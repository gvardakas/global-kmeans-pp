# The Global $k$-mean++ clustering algorithm

The $k$-means algorithm is a prevalent clustering method due to its simplicity, effectiveness, and speed. However, its main disadvantage is its high sensitivity to the initial positions of the cluster centers. The global $k$-means is a deterministic algorithm proposed to tackle the random initialization problem of k-means but its well-known that requires high computational cost. It partitions the data to $K$ clusters by solving all $k$-means sub-problems incrementally for all $k=1,\ldots, K$. For each $k$ cluster problem, the method executes the $k$-means algorithm $N$ times, where $N$ is the number of datapoints. The global $k$-means++ is an effective relaxation of the global $k$-means clustering algorithm, providing an ideal compromise between clustering error and execution speed. It is an effective way of acquiring quality clustering solutions akin to those of global $k$-means with a reduced computational load. It is an incremental clustering approach that dynamically adds one cluster center at each $k$ cluster sub-problem. For each $k$ cluster sub-problem, the method selects $L$ data points as candidates for the initial position of the new center using the effective $k$-means++ selection probability distribution. The selection method is fast and requires no extra computational effort for distance computations.

```
@article{vardakas2022global,
  title={Global $k$-means$++$: an effective relaxation of the global $k$-means clustering algorithm},
  author={Vardakas, Georgios and Likas, Aristidis},
  journal={arXiv preprint arXiv:2211.12271},
  year={2022}
}

@article{likas2003global,
  title={The global k-means clustering algorithm},
  author={Likas, Aristidis and Vlassis, Nikos and Verbeek, Jakob J},
  journal={Pattern recognition},
  volume={36},
  number={2},
  pages={451--461},
  year={2003},
  publisher={Elsevier}
}
```
