#Day 32: PageRank
<br>
https://en.wikipedia.org/wiki/PageRank
<br>
PageRank is an algorithm to determine what is called centrality in egonets. There are other ways to measure centrality, e.g. betweenness, closeness or hubs and authorities.
<br>
However, PageRank is quite useful for huge graphs due to its relation to Markov Chains. Specifically, if the graph adjacency matrix M is stochastic, irreducible and aperiodic, there exists a single point of convergence that is solution to equation M*r=r.
<br>
The solution is leading eigenvector and Power iteration is an iterative method that can be used to find it. Power iteration has fast convergence rate and its computation can be easily distributed.
<br>
map:
```
    let mut map = array![
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0,],
        [1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0,],
        [0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0,],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,],
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,],
        [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0,],
        [1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0,],
        [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0,],
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0,]
    ];

```

run:

```
    let rank = pagerank(&mut map, 0.9);
    assert_eq!(rank, array![0.09162540433565292, 0.16841237093737252, 0.05878810278583852, 0.022557218022556703, 0.029109244297414964, 0.11590770790575236, 0.22946066514576746, 0.13098324391698687, 0.055191867050503746, 0.09796417560215473]);

```