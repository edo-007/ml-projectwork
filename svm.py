
        case "svm":
            clf = LinearSVC(max_iter=5000)
            param_grid = {
                'C': [0.01, 0.1, 1, 10],
                'loss': ['hinge', 'squared_hinge']
            }

