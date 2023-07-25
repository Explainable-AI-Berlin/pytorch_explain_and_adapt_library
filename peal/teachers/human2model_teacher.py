import threading
import shutil
import os
import copy
import time

from flask import (
    Flask,
    render_template,
    request
)
from tqdm import tqdm

from peal.teachers.teacher_interface import TeacherInterface
from peal.utils import (
    get_project_resource_dir,
    is_port_in_use
)

class DataStore():
    i = None
    collage_paths = None
    feedback = None


class Human2ModelTeacher(TeacherInterface):
    '''
    
    '''
    def __init__(self, port):
        '''

        '''
        # TODO fix bug with reloading
        shutil.rmtree('static', ignore_errors = True)
        os.makedirs('static')
        shutil.rmtree('templates', ignore_errors = True)
        shutil.copytree(os.path.join(get_project_resource_dir(), 'templates'), 'templates')        
        self.port = port
        while is_port_in_use(self.port):
            print('port ' + str(self.port) + ' is occupied!')
            self.port += 1

        print('Start feedback loop!')
        #
        #host_name = "localhost"
        host_name = "0.0.0.0"
        app = Flask('feedback_loop')

        self.data = DataStore()
        self.data.i = 0
        self.data.collage_paths = []
        self.data.feedback = []

        app.config.UPLOAD_FOLDER = 'static'

        @app.route("/", methods=['GET', 'POST'])
        def index():
            if request.method == 'POST':
                if request.form['submit_button'] == 'True Counterfactual':
                    self.data.feedback.append('true')

                elif request.form['submit_button'] == 'False Counterfactual':
                    self.data.feedback.append('false')
                    
                elif request.form['submit_button'] == 'Out of Distribution':
                    self.data.feedback.append('ood')

                if len(self.data.collage_paths) > 0 and len(self.data.collage_paths) > self.data.i:
                    collage_path = self.data.collage_paths[self.data.i]
                    self.data.i += 1
                    return render_template('feedback_loop.html', form=request.form, counterfactual_collage = collage_path)

                else:
                    return render_template('information.html')

            elif request.method == 'GET':
                if len(self.data.collage_paths) > 0:
                    collage_path = self.data.collage_paths[self.data.i]
                    self.data.i += 1
                    return render_template('feedback_loop.html', form=request.form, counterfactual_collage = collage_path)

                else:
                    return render_template('information.html')

        self.thread = threading.Thread(target=lambda: app.run(host=host_name, port=self.port, debug=True, use_reloader=False))
        self.thread.start()
        print('Feedback GUI is active on localhost:' + str(self.port))

    def get_feedback(self, collage_paths, base_dir, **args):
        '''

        '''
        collage_paths_static = []
        for path in collage_paths:
            collage_path_static = os.path.join('static', path.split('/')[-1])
            shutil.copy(path, collage_path_static)
            collage_paths_static.append(collage_path_static)

        self.data.collage_paths = collage_paths_static

        with tqdm(range(100000)) as pbar:
            for it in pbar:
                if len(self.data.feedback) >= len(self.data.collage_paths):
                    break

                else:
                    pbar.set_description(
                        'Give feedback at localhost:' + str(self.port) + ', Current Feedback given: ' + str(len(self.data.feedback)) + '/' + str(len(self.data.collage_paths))
                    )
                    time.sleep(1.0)

        #stop_threads = True
        #thread.join()
        feedback = copy.deepcopy(self.data.feedback)
        self.data.collage_paths = []
        self.data.feedback = []
        self.data.i = 0
        return feedback