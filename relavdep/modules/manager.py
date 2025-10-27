import ray

@ray.remote
class Manager:
    def __init__(self):
        self.tasks = {}
        self.predictions = {}
        self.results = {}

    def get_tasks(self):
        return self.tasks

    def get_tasks_keys(self):
        return list(self.tasks.keys())
    
    def add_task(self, task, wt_seq, mut_seq):
        if task not in self.get_tasks():
            self.tasks[task] = [wt_seq, mut_seq]

    def remove_task(self, task):
        self.tasks.pop(task)

    def get_results(self):
        return self.results

    def get_results_keys(self):
        return list(self.results.keys())
    
    def get_result_item(self, task):
        return self.results[task]
    
    def save_result(self, task, fitness):
        self.results[task] = fitness

    def get_predictions(self):
        return list(self.predictions.keys())
    
    def get_prediction_item(self, seq):
        return self.predictions[seq]
    
    def save_prediction(self, seq, prediction):
        self.predictions[seq] = prediction
