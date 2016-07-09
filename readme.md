# Seyade document encoder

## Requirement
- Theano
- Lasagne (edge)
- NumPy
- SciPy

## Usage

Create directory under `models/` with its model name.

Place `train.txt`, `test.txt` and `encode.txt` under the directory.

### Train model

```bash
$ python3 train.py --model_name=mymodel
```

### Encode using trained model

```bash
$ python3 encode.py --model_name=mymodel
```

### See prediction result

```python
from shared import *
model = Seyade('qiita_8k')
model.load_result()

doc = model.train_docs[0]
doc['text'] # => 'Lorem ipsum ...'
doc['targets'] # => ['foo', 'baz']
doc['predictions'] # => ['foo', 'bar']
doc['scored_predictions'] # => [('foo', 0.95), ('bar', 0.88)]
doc['embedding'] # => array([ 0.1,  0.5, -0.9, ...])

for doc in model.encode_docs:
  print(doc['targets'], doc['scored_predictions'], doc['text'])

# comparable precision@1 score with old method
model.load_result(0)
correct = 0
for doc in model.encode_docs:
  if doc['predictions'][0] in doc['targets']:
    correct += 1

# Set different threshold for prediction.
model.load_result(0.9) # 0.7 => 82.61(632/765), 0.9 => 87.66, 0.99 => 93.48
correct = 0
total = 0
for doc in model.test_docs:
  for p in doc['predictions']:
    total += 1
    if p == doc['targets'][0]:
      correct += 1

print(total, correct, correct / total)
```
