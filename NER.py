import spacy
#from __future__ import unicode_literals, print_function
from tqdm import tqdm # loading bar
import fire
import random
from pathlib import Path


#nlp = spacy.load('en')

#sent_0 = nlp(u'Donald Trump visited at the government headquarters in France today.')

#sent_1 = nlp(u'Emmanuel Jean-Michel Frédéric Macron is a French politician serving as President of France and ex officio Co-Prince of Andorra since 14 May 2017.')

#sent_2 = nlp(u'He studied philosophy at Paris Nanterre University, completed a Master’s of Public Affairs at Sciences Po, and graduated from the École nationale d\'administration (ÉNA) in 2004. ')

#sent_3 = nlp(u'He worked at the Inspectorate General of Finances, and later became an investment banker at Rothschild & Cie Banque.')

#for token in sent_0:
#    print(token.text, token.ent_type_)


#for ent in sent_0.ents:
#    print(ent.text, ent.label_)

#for token in sent_1:
#    print(token.text, token.ent_type_)

#for ent in sent_1.ents:
#    print(ent.text, ent.label_)

#for token in sent_2:
#    print(token.text, token.ent_type_)

#for ent in sent_2.ents:
#    print(ent.text, ent.label_)

#for token in sent_3:
#    print(token.text, token.ent_type_)

#for ent in sent_3.ents:
#    print(ent.text, ent.label_)


# run this code seperately


# training data
TRAIN_DATA = [
    ('Who is Shaka Khan?', {
        'entities': [(7, 17, 'PERSON')]
    }),
    ('I like London and Berlin.', {
        'entities': [(7, 13, 'LOC'), (18, 24, 'LOC')]
    })
]


def main(model=None, output_dir=None, n_iter=100):
    """Load the model, set up the pipeline and train the entity recognizer."""
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank('en')  # create blank Language class
        print("Created blank 'en' model")

    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner, last=True)
    # otherwise, get it so we can add labels
    else:
        ner = nlp.get_pipe('ner')

    # add labels
    for _, annotations in TRAIN_DATA:
        for ent in annotations.get('entities'):
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        optimizer = nlp.begin_training()
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            losses = {}
            for text, annotations in TRAIN_DATA:
                nlp.update(
                    [text],  # batch of texts
                    [annotations],  # batch of annotations
                    drop=0.5,  # dropout - make it harder to memorise data
                    sgd=optimizer,  # callable to update weights
                    losses=losses)
            print(losses)

    # test the trained model
    for text, _ in TRAIN_DATA:
        doc = nlp(text)
        print('Entities', [(ent.text, ent.label_) for ent in doc.ents])
        print('Tokens', [(t.text, t.ent_type_, t.ent_iob) for t in doc])

    # save model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        for text, _ in TRAIN_DATA:
            doc = nlp2(text)
            print('Entities', [(ent.text, ent.label_) for ent in doc.ents])
            print('Tokens', [(t.text, t.ent_type_, t.ent_iob) for t in doc])




if __name__ == '__main__':
  fire.Fire()
