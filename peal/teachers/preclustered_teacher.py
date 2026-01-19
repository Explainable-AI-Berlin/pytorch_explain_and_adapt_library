from peal.teachers.interfaces import TeacherInterface


class PreclusteredTeacher(TeacherInterface):
    def __init__(self, dataset, correct_clusters, tracking_level=0, counterfactual_type="1sided"):
        self.dataset = dataset
        self.tracking_level = tracking_level
        self.counterfactual_type = counterfactual_type
        self.correct_clusters = correct_clusters

    def get_feedback(
        self,
        x_counterfactual_list,
        y_source_list,
        x_list,
        y_list,
        y_target_end_confidence_list,
        cluster_list,
        base_dir=None,
        y_target_list=None,
        mode="train",
        **kwargs
    ):
        feedback = []
        teacher_original = []
        teacher_counterfactual = []
        for idx, counterfactual in enumerate(x_counterfactual_list):
            outlier_score = float(self.dataset.calculate_outlier_score(counterfactual.unsqueeze(0))['relative'])

            if (
                self.counterfactual_type == "1sided"
                and y_list[idx] != y_source_list[idx]
            ):
                feedback.append("student originally wrong!")

            elif y_target_end_confidence_list[idx] < 0.5:
                feedback.append("student not swapped!")

            elif outlier_score > 2.0:
                feedback.append("ood_" + str(round(outlier_score, 2)))

            else:
                if cluster_list[idx] in self.correct_clusters:
                    feedback.append("true")

                else:
                    feedback.append("false")

            teacher_original.append(-1)
            teacher_counterfactual.append(-1)

        if (
            self.tracking_level >= 5
            and mode == "validation"
            or self.tracking_level >= 5
            and mode == "train"
        ):
            self.dataset.generate_contrastive_collage(
                y_counterfactual_teacher_list=teacher_counterfactual,
                y_original_teacher_list=teacher_original,
                feedback_list=feedback,
                x_counterfactual_list=x_counterfactual_list,
                y_source_list=y_source_list,
                y_target_list=y_target_list,
                x_list=x_list,
                y_list=y_list,
                y_target_end_confidence_list=y_target_end_confidence_list,
                base_path=base_dir,
                **kwargs,
            )

        return feedback
