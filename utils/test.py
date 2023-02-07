from pydprint import dprint as dp
def show_random_elements(dataset, num_examples=5):
    from IPython.display import display, HTML
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)
    
    df = pd.DataFrame(dataset[picks])
    for column, typ in dataset.features.items():
        if isinstance(typ, datasets.ClassLabel):
            df[column] = df[column].transform(lambda i: typ.names[i])
    display(HTML(df.to_html()))


def show_metric(m):
    fake_preds = ["hello there", "general kenobi"]
    fake_labels = ["hello there", "general kenobi"]
    dp(m.compute(predictions=fake_preds, references=fake_labels))

def show_tokenizer(t):
    test_sentences = ["Hello, this is one sentence!", "This is another sentence."]
    dp(t(test_sentences))

    with t.as_target_tokenizer(): dp(t(test_sentences))
