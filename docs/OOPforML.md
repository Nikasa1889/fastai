A Machine Learning pipeline usually consists of multiple steps, such as loading data, transforming data, randomizing data, splitting data,
batching data, building a model, fitting a model, logging fitting process, reloading a model, predicting...
Fastai library structures a ML pipeline by objectifing its steps in a hierarchical manner (Object-Oriented Programming).

This is one of the two most common ways to build up an ML pipeline, which indeed are usually mixed together in production.
- Pipeline the Make way (similar to GNU Make, pipeline of scikit-learn or spotify luigi).
  + Steps are chained together depending on their data dependency
  + One step is either a transformer (transform a data from input to output) or estimator (a transformer but can learn the transformation)
  + All steps are equal, no matter how small or big they are. They either can predict() or fit() and predict().
  + A pipeline is created by defining all steps and chain them by their dependency order.
  + A pipeline can be updated by replacing a step by another equivalent, which is only known by the programmer.
  
- Pipeline the OOP way (the Pytorch and fastai way).
  + A step is modelled as an object that knows how to do some job.
  + Related steps can be grouped to form a bigger step.
  + Steps are not equal. A step implements a specific protocol depending on its role.
  + The bigger step can orchestrate its smaller steps in different way to get its job done.
  + Ideally, in the end, the whole pipeline is represented by a single biggest step.
  + The pipeline is created by using dependency injection or factory methods. Small steps get defined first, then inject to
  the constructor of the bigger steps. Or if the constructing process is complicated and usually happen in the same way, fastai
  provide a factory method to easily construct a big step right away.
  + A pipeline can be updated by replacing steps that implementing the same protocol.

We see that, the Make pipeline is a linear execuation graph where output of one step is fed into input of the next one.
With OOP approach, the big step can excecute its small steps in whatever order it needs, which can even be looping or branching.
In this doc, we explore the fastai approach, especially focus on how fastai breaks up the pipeline into steps.
Each step has a protocol that you must follow if you want to implement a the step yourself.

## Fastai ML components:
- Data Pipeline
  + Dataset
  + Shuffler
  + BatchShuffler
  + Dataloader
  + Transformer
  + Tokenizer
  
- Model
  + Model
  + ModelBuilder (ConvnetBuilder)
  + Initializer
  + Layer_learner
  + Optimizer
  
- Objective
  + Loss
  + Metric
  + Plot
  
- Whole pipeline
  + Learner
