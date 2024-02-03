clean:
  @rm -rf weight test-result.txt

modeltrain:
  @python3 train.py train

modeltest:
  @python3 train.py test

test:
  @yarn test
