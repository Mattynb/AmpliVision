import pickle as pkl


def save_results(Eval):
    with open(r'Eval.pkl', 'wb') as f:
        pkl.dump(Eval, f)


def load_results():
    with open(r'Eval.pkl', 'rb') as f:
        Eval = pkl.load(f)

    return Eval
