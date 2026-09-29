"""Interactive Flask teacher that asks a human to label counterfactual collages.

PEAL teachers turn counterfactual explanations into feedback for the adaptor.
:class:`Human2ModelTeacher` serves each collage in a small web page (templates
``feedback_loop.html`` / ``information.html``) and blocks in
:meth:`Human2ModelTeacher.get_feedback` until the person has answered
"True Counterfactual", "False Counterfactual" or "Out of Distribution" for
every image. Collages are copied into a local ``static`` folder for serving.
"""

import tempfile
import threading
import shutil
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
    """Mutable state shared between the Flask route and the training thread.

    Attributes
    ----------
    i : int
        Index of the next collage to show.
    collage_paths : list of str
        Paths under ``static/`` of the collages awaiting feedback.
    feedback : list of str
        Collected answers (``"true"``, ``"false"``, ``"ood"`` or an automatic
        ``"student incorrect!"`` / ``"student not swapped!"`` entry).
    y_target_end_confidence_list : list of float
        Confidence of the student in the target class after the edit.
    student_correct_list : list of bool
        Whether the student's source prediction matched the label.
    """

    i = None
    collage_paths = None
    feedback = None
    y_target_end_confidence_list = None
    student_correct_list = None


class Human2ModelTeacher(TeacherInterface):
    """Teacher whose feedback comes from a human through a local web form.

    Parameters
    ----------
    port : int
        Preferred port for the Flask app; incremented until a free port is found.

    Notes
    -----
    The constructor wipes and recreates a ``static`` directory in the current
    working directory, starts the Flask server on ``0.0.0.0`` in a daemon-less
    background thread and never stops it. Collages whose student prediction is
    wrong or whose target confidence stays below 0.5 are skipped and answered
    automatically instead of being shown.
    """

    def __init__(self, port):
        """Create the ``static`` folder, pick a free port and start the server.

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

        @app.route("/", methods=["GET", "POST"])
        def index():
            """Record the submitted answer and render the next collage."""
            if request.method == "POST":
                if request.form["submit_button"] == "True Counterfactual":
                    self.data.feedback.append("true")

                elif request.form["submit_button"] == "False Counterfactual":
                    self.data.feedback.append("false")

                elif request.form["submit_button"] == "Out of Distribution":
                    self.data.feedback.append("ood")

                while (
                    self.data.y_target_end_confidence_list[self.data.i] < 0.5
                    or not self.data.student_correct_list[self.data.i]
                ):
                    if not self.data.student_correct_list[self.data.i]:
                        self.data.feedback.append("student incorrect!")

                    if self.data.y_target_end_confidence_list[self.data.i] < 0.5:
                        self.data.feedback.append("student not swapped!")

                    self.data.i += 1
                    if self.data.i >= len(self.data.collage_paths):
                        break

                if (
                    len(self.data.collage_paths) > 0
                    and len(self.data.collage_paths) > self.data.i
                ):
                    collage_path = self.data.collage_paths[self.data.i]
                    self.data.i += 1
                    return render_template(
                        "feedback_loop.html",
                        form=request.form,
                        counterfactual_collage=collage_path,
                    )

                else:
                    return render_template("information.html")

            elif request.method == "GET":
                if len(self.data.collage_paths) > 0:
                    collage_path = self.data.collage_paths[self.data.i]
                    self.data.i += 1
                    return render_template(
                        "feedback_loop.html",
                        form=request.form,
                        counterfactual_collage=collage_path,
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

    def get_feedback(
        self,
        collage_path_list,
        y_target_end_confidence_list,
        y_source_list,
        y_list,
        **kwargs,
    ):
        """Show every collage to the human and wait for one answer each.

        Parameters
        ----------
        collage_path_list : list of str
            Collage images; each is copied to ``static/<basename>``.
        y_target_end_confidence_list : list of float
            Student confidence in the target class after the counterfactual edit.
        y_source_list : list
            Student predictions on the original samples.
        y_list : list
            Ground-truth labels; compared with ``y_source_list`` to mark samples
            the student got wrong.
        **kwargs
            Ignored; present for interface compatibility with other teachers.

        Returns
        -------
        list of str
            One entry per collage: ``"true"``, ``"false"``, ``"ood"``,
            ``"student incorrect!"`` or ``"student not swapped!"``. Polls once a
            second (up to 100000 times) until the list is complete, then resets
            the shared :class:`DataStore`.
        """
        _log.info("%s", "start collecting feedback!!!")
        collage_paths_static = []
        for path in collage_path_list:
            # Copy into the served directory; the template gets the URL path,
            # which Flask maps onto static_folder.
            name = path.split("/")[-1]
            shutil.copy(path, os.path.join(self.static_dir, name))
            collage_paths_static.append(os.path.join("static", name))

        self.data.collage_paths = collage_paths_static
        self.data.y_target_end_confidence_list = y_target_end_confidence_list
        self.data.student_correct_list = [
            y_source_list[i] == y_list[i] for i in range(len(y_list))
        ]

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

        # stop_threads = True
        # thread.join()
        feedback = copy.deepcopy(self.data.feedback)
        self.data.collage_paths = []
        self.data.feedback = []
        self.data.i = 0
        return feedback
