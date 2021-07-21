---
title: Deep Neural Network
author: Sidharth Baskaran
date: July 2021
graphics: true
header-includes:
- \graphicspath{{./images/}}
---

# Deep L-layer neural network

* Logistic regression is shallow model, and a deeper network has more hidden layers
* Notation
  * $L$ - number of layers in network
  * $n^{[l]}$ - number of units in layer $l$
    * $n^{[0]}=n_x$
  * $a^{[l]}=g^{[l]}(z^{[l]})$
    * $a^{[0]}=x,a^{[L]}=\hat{y}$

# Forward propagation

* Steps
  * $z^{[l]}=w^{[l]}a^{[l-1]}+b^{[l]}$
  * $a^{[l]}=g^{[l]}(z^{[l]})$
* Vectorized across $m$ examples
  * $Z^{[l]}=W^{[l]}A^{[l-1]}+b^{[l]}$ where $X=A^{[0]}$
  * $Z,A,X$ are stacked columnwise, i.e. $Z^{[1](1)},\ldots,Z^{[L](m)}$

# Matrix Dimension Debugging

* Forward propagation step

$$
\begin{aligned}
z^{[l]}&=W^{[l]}a^{[l-1]}+b^{[l]}\\
(n^{[l]},1)&=(n^{[l]},n^{[l-1]})(n^{[l-1]},1)+(n^{[l]},1)
\end{aligned}
$$

* If vectorized, must modify

$$
\begin{aligned}
Z^{[l]}&=W^{[l]}A^{[l-1]}+b^{[l]}\\
(n^{[l]},m)&=(n^{[l]},n^{[l-1]})(n^{[l-1]},m)+\underbrace{(n^{[l]},1)}_{\text{broadcasted}}
\end{aligned}
$$