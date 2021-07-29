---
title: Optimization Algorithms
author: Sidharth Baskaran
date: July 2021
graphics: true
header-includes:
- \graphicspath{{./images/}}
---

# Mini-batch gradient descent

- Vectorization allows for compute on $m$ examples
    - Let $X=[ x^{(1)},\ldots, x^{(1000)}|x^{(1001)},\ldots,x^{(2000)}]$ be split into $x^{\{1\}}$ and $x^{\{2\}}$ for example, these are the batches
    - Up to 5000 batches
    - $Y$ can also be divided this way into minibatches
    - $X^{\{j\}}$ has dimension $(n_x,t)$ and $Y^{\{j\}}$ is of $(1,t)$
        - $t$ is the batch size
- Use vectorization to process
    - For each minibatch, perform propagation step using each minibatch
    - Can then calculate cost and perform backprop
- Epoch is a single pass through training set

# Understanding mini-batch gradient descent
