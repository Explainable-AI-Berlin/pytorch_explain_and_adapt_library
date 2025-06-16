import threading
import shutil
import os
import copy
import time

from flask import Flask, render_template, request
from tqdm import tqdm

from peal.teachers.interfaces import TeacherInterface
from peal.global_utils import get_project_resource_dir, is_port_in_use


class DataStore:
    i = None
    collage_paths = None
    feedback = None


class ClusterTeacher(TeacherInterface):
    """ """

    def __init__(self, port, dataset, tracking_level=0, counterfactual_type="1sided"):
        """ """
        # TODO fix bug with reloading
        shutil.rmtree("static", ignore_errors=True)
        self.dataset = dataset
        self.tracking_level = tracking_level
        os.makedirs("static")
        self.port = port
        while is_port_in_use(self.port):
            print("port " + str(self.port) + " is occupied!")
            self.port += 1

        print("Start feedback loop!")
        #
        # host_name = "localhost"
        host_name = "0.0.0.0"
        app = Flask("feedback_loop")

        self.data = DataStore()
        self.data.i = 0
        self.data.collage_paths = []
        self.data.feedback = []

        app.config.UPLOAD_FOLDER = "static"
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

                if len(self.data.collage_paths) > 0 and len(self.data.collage_paths) > self.data.i:
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
            target=lambda: app.run(host=host_name, port=self.port, debug=True, use_reloader=False)
        )
        self.thread.start()
        print("Feedback GUI is active on localhost:" + str(self.port))

    def get_feedback(self, num_clusters, **kwargs):
        """ """
        print("start collecting feedback!!!")
        collage_path_clusters = []
        l = len(kwargs["collage_path_list"]) // num_clusters
        for cluster_idx in range(num_clusters):
            collage_path_clusters.append(kwargs["collage_path_list"][cluster_idx * l : (cluster_idx + 1) * l])

        collage_clusters_static = []
        for collage_path_list in collage_path_clusters:
            collage_paths_static = []
            for path in collage_path_list:
                collage_path_static = os.path.join("static", path.split("/")[-1])
                shutil.copy(path, collage_path_static)
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
            if self.counterfactual_type == "1sided" and kwargs["y_list"][idx] != kwargs["y_source_list"][idx]:
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
