# Inductive Bias Experiment
By Jong M. Shin et al at Johns Hopkins University
> This is still work in progress

# Conference
The work has been [presented](posters/NAISys2022-poster.pdf) at [NAISys 2022](https://meetings.cshl.edu/meetings.aspx?meet=NAISYS&year=22) on April 5-9 at Cold Spring Harbor Laboratory in New York, USA.

# Introduction
The neural network’s popularity as a universal approximator has relatively been a recent phenomenon since the first artificial intelligence (AI) winter. Its initial motivation as a model of human intelligence dramatically failed a sanity check by a simple nonlinearity 60 years ago. With the advent of deeper architecture to imitate biological neural connection, this concept has regained traction as a promising mode of creating machine intelligence. Despite a huge leap from a single layer perceptron, however, deep neural networks have also been subjected to inductive bias. In many cases, its ability to extrapolate falls behind its counterpart algorithms, and oftentimes, its performance is not even on par with that of humans.

Many notable AI scientists are concerned about the possible return of AI winter due to these overinflated expectations that deep learning is an omnipotent methodology to bring about true machine intelligence. One of the very reasons for the first AI winter was the similarly ungrounded hype that had been a good idea but was ahead of time. Current study aims to critically assess various deep learning methods to raise awareness of inductive bias coinciding with machine learning, and ultimately propose a solution to minimize this corollary effect.

# Experiments
In order to investigate inductive bias associated with machine learning models, I trained six different algorithms on five datasets generated from two distinct patterns. I employed a multi-layer perceptron as a model of deep nets in contrast to the other machine learning models such as random forest, support vector machine, k-nearest neighbor, and quadratic discriminant analysis. I subsequently compared estimated posterior probability predicted by these models and computed Hellinger distances as an evaluation metric with respect to the corresponding true posterior probability. As a result, all of these models were better interpolators than they were extrapolators. Notably, deep nets performed worst across all dataset particularly with a spiral pattern while random forest single-handedly performed the best. This is a clear demonstration of bias inherent to machine learning, and a case-study that deep nets may not be the most representative model for human intelligence.

As a proposed model of human intelligence, it was also imperative to benchmark the performance of deep nets relative to the human outputs. To achieve this goal, I have developed a web application to simulate machine learning processes to train virtual human operators from Amazon Mechanical Turk. This application is currently undergoing a final review process by our collaborators before its full deployment. Along with the findings from previously discussed experiments, I believe this behavioral experiment will serve as strong evidence to attract attention from the deep learning community.

# Log
Log can be viewed from [here](LOG.md).