"""Human-in-the-loop teacher that collects one verdict per counterfactual cluster.

A CFKD teacher answers "true", "false" or "ood" for every counterfactual the
explainer produced. This teacher serves the pre-rendered collages of each
cluster through a small Flask page (``clustered_feedback_loop.html``), waits
until a person has clicked a verdict for every cluster and broadcasts that
verdict to all counterfactuals of the cluster. It is the interactive
counterpart of the automatic teachers in ``peal.teachers``.
"""

import threading
import shutil
import tempfile
import os
import copy
import time

from tqdm import tqdm

from peal._optional import require
from peal.teachers.interfaces import TeacherInterface
from peal.global_utils import is_port_in_use
from peal.log import get_logger

_log = get_logger(__name__)


class DataStore:
    """Mutable state shared between the Flask request handler and the teacher.

    Attributes
    ----------
    i : int
        Index of the next cluster to show.
    collage_paths : list of list of str
        One list of collage paths (under ``static/``) per cluster.
    feedback : list of str
        Verdicts collected so far, one per cluster, in display order.
    """

    i = None
    collage_paths = None
    feedback = None


class ClusterTeacher(TeacherInterface):
    """Teacher that asks a human for one verdict per cluster of counterfactuals.

    The constructor wipes and recreates a ``static`` directory in the current
    working directory, starts a Flask app on a daemon thread bound to
    ``0.0.0.0`` and the first free port at or above ``port``, and keeps a
    :class:`DataStore` that the request handler and :meth:`get_feedback` share.

    Parameters
    ----------
    port : int
        First port to try; incremented until a free one is found.
    dataset : peal.data.datasets.ImageDataset
        Used to render contrastive collages when ``tracking_level >= 5``.
    tracking_level : int
        Verbosity/artifact level of the surrounding CFKD run.
    counterfactual_type : str
        ``"1sided"`` marks counterfactuals whose factual was misclassified as
        ``"student originally wrong!"`` instead of a verdict.
    """

    def __init__(self, port, dataset, tracking_level=0, counterfactual_type="1sided"):
        """Start the feedback web server; see the class docstring.

        Raises
        ------
        ImportError
            If the optional ``flask`` dependency is not installed.
        """
        flask = require("flask", "web", "the interactive feedback web app")
        Flask = flask.Flask
        render_template = flask.render_template
        request = flask.request

        # A per-instance temporary directory for the collages the browser is
        # served. This used to delete a *relative* ``static`` folder, i.e. one
        # in whatever the caller's working directory happened to be - a library
        # must not do that.
        self.static_dir = tempfile.mkdtemp(prefix="peal_feedback_")
        self.dataset = dataset
        self.tracking_level = tracking_level
        self.port = port
        while is_port_in_use(self.port):
            _log.info("%s", "port " + str(self.port) + " is occupied!")
            self.port += 1

        _log.info("%s", "Start feedback loop!")
        #
        # host_name = "localhost"
        host_name = "0.0.0.0"
        app = Flask(
            "feedback_loop", static_folder=self.static_dir, static_url_path="/static"
        )

        self.data = DataStore()
        self.data.i = 0
        self.data.collage_paths = []
        self.data.feedback = []

        app.config.UPLOAD_FOLDER = self.static_dir
        self.counterfactual_type = counterfactual_type

        @app.route("/", methods=["GET", "POST"])
        def index():
            if request.method == "POST":
                if request.form["submit_button"] == "True Counterfactual":
                    self.data.feedback.append("true")

                elif request.form["submit_button"] == "False Counterfactual":
                    self.data.feedback.append("false")

                elif request.form["submit_button"] == "Out of Distribution":
                    self.data.feedback.append("ood")

                if (
                    len(self.data.collage_paths) > 0
                    and len(self.data.collage_paths) > self.data.i
                ):
                    collage_path = self.data.collage_paths[self.data.i]
                    self.data.i += 1
                    return render_template(
                        "clustered_feedback_loop.html",
                        form=request.form,
                        counterfactual_collages=collage_path,
                    )

                else:
                    return render_template("information.html")

            elif request.method == "GET":
                if len(self.data.collage_paths) > 0:
                    collage_path = self.data.collage_paths[self.data.i]
                    self.data.i += 1
                    return render_template(
                        "clustered_feedback_loop.html",
                        form=request.form,
                        counterfactual_collages=collage_path,
                    )

                else:
                    return render_template("information.html")

        self.thread = threading.Thread(
            target=lambda: app.run(
                host=host_name, port=self.port, debug=True, use_reloader=False
            )
        )
        self.thread.start()
        _log.info("%s", "Feedback GUI is active on localhost:" + str(self.port))

    def get_feedback(self, num_clusters, **kwargs):
        """Block until a person has judged every cluster and expand the verdicts.

        ``collage_path_list`` is split into ``num_clusters`` equally sized
        consecutive chunks, the collages are copied into ``static/`` and served
        one cluster at a time. The method polls once per second until as many
        verdicts as clusters have been submitted, then repeats each cluster's
        verdict for every counterfactual in that cluster.

        Parameters
        ----------
        num_clusters : int
            Number of clusters the counterfactuals are grouped into.
        **kwargs
            The CFKD ``tracked_values``: ``collage_path_list``,
            ``x_counterfactual_list``, ``x_list``, ``y_list``,
            ``y_source_list``, ``y_target_list``,
            ``y_target_start_confidence_list``,
            ``y_target_end_confidence_list`` and ``base_dir``.

        Returns
        -------
        list of str
            One entry per counterfactual: ``"true"``, ``"false"`` or ``"ood"``
            from the human, overridden by ``"student originally wrong!"``
            (1-sided mode, factual misclassified) or ``"student not swapped!"``
            (target confidence below 0.5). At ``tracking_level >= 5`` a
            contrastive collage is additionally written to ``base_dir``.
        """
        _log.info("%s", "start collecting feedback!!!")
        collage_path_clusters = []
        l = len(kwargs["collage_path_list"]) // num_clusters
        for cluster_idx in range(num_clusters):
            collage_path_clusters.append(
                kwargs["collage_path_list"][cluster_idx * l : (cluster_idx + 1) * l]
            )

        collage_clusters_static = []
        for collage_path_list in collage_path_clusters:
            collage_paths_static = []
            for path in collage_path_list:
                # Copy into the served directory; the list handed to the template
                # holds the URL path, which Flask maps onto static_folder.
                name = path.split("/")[-1]
                shutil.copy(path, os.path.join(self.static_dir, name))
                collage_path_static = os.path.join("static", name)
                collage_paths_static.append(collage_path_static)

            collage_clusters_static.append(collage_paths_static)

        self.data.collage_paths = collage_clusters_static

        with tqdm(range(100000)) as pbar:
            for it in pbar:
                if len(self.data.feedback) >= len(self.data.collage_paths):
                    break

                else:
                    pbar.set_description(
                        "Give feedback at localhost:"
                        + str(self.port)
                        + ", Current Feedback given: "
                        + str(len(self.data.feedback))
                        + "/"
                        + str(len(self.data.collage_paths))
                    )
                    time.sleep(1.0)

        feedback = copy.deepcopy(self.data.feedback)
        self.data.collage_paths = []
        self.data.feedback = []
        self.data.i = 0
        feedback_out = []
        for cluster_idx in range(num_clusters):
            for _ in range(l):
                feedback_out.append(feedback[cluster_idx])

        for idx, counterfactual in enumerate(kwargs["x_counterfactual_list"]):
            if (
                self.counterfactual_type == "1sided"
                and kwargs["y_list"][idx] != kwargs["y_source_list"][idx]
            ):
                feedback_out[idx] = "student originally wrong!"

            elif kwargs["y_target_end_confidence_list"][idx] < 0.5:
                feedback_out[idx] = "student not swapped!"

        if self.tracking_level >= 5:
            self.dataset.generate_contrastive_collage(
                y_counterfactual_teacher_list=[-1] * len(feedback_out),
                y_original_teacher_list=[-1] * len(feedback_out),
                feedback_list=feedback_out,
                x_counterfactual_list=kwargs["x_counterfactual_list"],
                y_source_list=kwargs["y_source_list"],
                y_target_list=kwargs["y_target_list"],
                x_list=kwargs["x_list"],
                y_list=kwargs["y_list"],
                y_target_end_confidence_list=kwargs["y_target_end_confidence_list"],
                y_target_start_confidence_list=kwargs["y_target_start_confidence_list"],
                base_path=kwargs["base_dir"],
            )

        return feedback_out
