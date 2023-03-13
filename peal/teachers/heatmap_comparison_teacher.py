import torch
import numpy as np

from PIL import Image

from peal.teachers.teacher_interface import TeacherInterface


class HeatmapComparisonTeacher(TeacherInterface):
    def __init__(self, explainer, attribution_threshold):
        self.explainer = explainer
        self.attribution_threshold = attribution_threshold

    def get_feedback(self, images, source_classes, predictions, heatmaps, **args):
        # TODO why not simply compare the predictions?
        feedback = []
        for idx, heatmap in enumerate(heatmaps):
            #
            if source_classes[idx] != predictions[idx]:
                feedback.append('false')
                
            else:
                heatmap_oracle = self.explainer.explain_batch(images[idx].unsqueeze(0), torch.tensor([source_classes[idx]]))[0].squeeze(0)
                #normalized_heatmap = heatmap.sum(0) / heatmap.sum()
                #normalized_hints = hints.sum(0) / hints.sum()
                # TODO wasserstein distance???
                #attribution_relative = (heatmap * hints).mean()
                #masked_pixels_relative = hints[idx].sum() / torch.prod(torch.tensor(list(hints[idx].shape)))
                #attribution_relative = attribution_relative / masked_pixels_relative
                img = Image.fromarray(np.array(255 * torch.cat([heatmap, heatmap_oracle], 2).numpy().transpose(1,2,0), dtype= np.uint8))
                heatmap = heatmap / heatmap.sum()
                heatmap_oracle = heatmap_oracle / heatmap_oracle.sum()
                attribution_relative = torch.nn.CosineSimilarity()(heatmap.sum(0).flatten().unsqueeze(0), heatmap_oracle.sum(0).flatten().unsqueeze(0))
                img.save('tmp' + str(idx) + '_' + str(int(1000 * attribution_relative)) + '.png')
                # TODO manually inspect some of these similarities
                if idx == 10:
                    from IPython.core.debugger import set_trace
                    set_trace()
                #
                if attribution_relative > self.attribution_threshold:
                    feedback.append('true')

                else:
                    feedback.append('false')

        return feedback
