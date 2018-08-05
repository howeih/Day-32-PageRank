#[macro_use(array)]
extern crate ndarray;
use ndarray::Array1;
use ndarray::Array2;

fn remove_self_link(graph: &mut Array2<f64>, dim: usize) {
    for i in 0..dim {
        graph[(i, i)] = 0.
    }
}

fn ensure_stochasticity(graph: &mut Array2<f64>, dim: usize) {
    for c in 0..dim {
        let mut sum = 0.;
        for r in 0..dim {
            sum += graph[(r, c)]
        }
        if sum == 0. {
            for r in 0..dim {
                graph[(r, c)] = 1.;
                sum += 1.;
            }
        }
        for r in 0..dim {
            graph[(r, c)] = graph[(r, c)] / sum;
        }
    }
}

fn add_random_teleports(graph: &mut Array2<f64>, dim: usize, alpha: f64) {
    for c in 0..dim {
        for r in 0..dim {
            graph[(r, c)] = graph[(r, c)] * alpha + (1. - alpha) / dim as f64;
        }
    }
}

fn init_rank(prev: &Array1<f64>, dim: usize) -> Array1<f64> {
    let mut rank = Array1::<f64>::zeros((dim,));
    for i in 0..dim {
        rank[i] = prev[i] + 1. / dim as f64;
    }
    rank
}

fn pagerank(graph: &mut Array2<f64>, alpha: f64) -> Array1<f64> {
    let graph_dim = graph.raw_dim();
    assert_eq!(graph_dim[0], graph_dim[1]);
    let dim = graph_dim[0];
    remove_self_link(graph, dim);
    ensure_stochasticity(graph, dim);
    add_random_teleports(graph, dim, alpha);
    let mut prev = Array1::<f64>::zeros((dim,));
    let mut rank = init_rank(&prev, dim);
    while (&rank - &prev).dot(&(&rank - &prev)) > 1e-8 {
        prev = rank;
        rank = graph.dot(&prev);
    }
    rank
}

fn main() {
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
    let rank = pagerank(&mut map, 0.9);
    assert_eq!(rank, array![0.09162540433565292, 0.16841237093737252, 0.05878810278583852, 0.022557218022556703, 0.029109244297414964, 0.11590770790575236, 0.22946066514576746, 0.13098324391698687, 0.055191867050503746, 0.09796417560215473]);
}
