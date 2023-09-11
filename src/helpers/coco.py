class Coco:
    def __init__():
        self.results = []
        self.correct = 0
        self.total_samples = 0
        self.ids = []

    def add(self, results):
        self.results.extend(results)
    
    def __call__(self, results, ids, expected=None, result_dict=None):
        processed = []
        for i in range(0, results[0]):
            self.ids.append(ids[i])
            processed.append([])
            detection_num = int(results[0][i])
            detection_boxes = results[1][i]
            detection_classes = expected[3][i]
            expected_classes = expected[i][0]

