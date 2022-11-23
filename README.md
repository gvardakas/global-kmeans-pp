# The Global $k$-mean++ clustering algorithm

The global $k$-means++ is an effective relaxation of the global $k$-means clustering algorithm, providing an ideal compromise between clustering error and execution speed. It is an effective way of acquiring quality clustering solutions akin to those of global $k$-means with a reduced computational load. It is an incremental clustering approach that dynamically adds one cluster center at each $k$ cluster sub-problem. For each $k$ cluster sub-problem, the method selects $L$ data points as candidates for the initial position of the new center using the effective $k$-means++ selection probability distribution. The selection method is fast and requires no extra computational effort for distance computations.

```
@misc{https://doi.org/10.48550/arxiv.2211.12271,
  doi = {10.48550/ARXIV.2211.12271},
  url = {https://arxiv.org/abs/2211.12271},
  author = {Vardakas, Georgios and Likas, Aristidis},
  keywords = {Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Global $k$-means$++$: an effective relaxation of the global $k$-means clustering algorithm},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
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
