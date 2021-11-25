use linfa::DatasetBase;
use linfa::traits::{Fit, Transformer};
use linfa_tsne::Result;
use mnist::{Mnist, MnistBuilder};
use ndarray::{ArrayBase, OwnedRepr, Dim};
use std::{io::Write, process::Command};

fn export_data(ds: DatasetBase<ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>, ArrayBase<OwnedRepr<u8>, Dim<[usize; 2]>>>) {
    let mut f = std::fs::File::create("data/mnist.dat").unwrap();

    println!("write to file");

    for (x, y) in ds.sample_iter() {
        f.write_all(format!("{} {} {}\n", x[0], x[1], y[0]).as_bytes())
            .unwrap();
    }

    Command::new("gnuplot")
        .arg("-p")
        .arg("data/mnist_plot.plt")
        .spawn()
        .expect(
            "Failed to launch gnuplot. Pleasure ensure that gnuplot is installed and on the $PATH.",
        );
} 

// https://github.com/rust-ml/linfa/blob/master/algorithms/linfa-tsne/examples/mnist.rs
fn main() -> Result<()> {
    // use 50k samples from the MNIST dataset
    let trn_size: usize = 5000;
    let tst_size: usize = 100;
    let (rows, cols) = (28, 28);

    println!("start!");

    println!("downloading dataset...");
    let Mnist {
        trn_img, trn_lbl, tst_img, tst_lbl, ..
    } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(trn_size as u32)
        .test_set_length(tst_size as u32)
        .download_and_extract()
        .finalize();

    println!("preparing dataset");
    let ds = linfa::Dataset::new(
        ndarray::Array::from_shape_vec((trn_size, rows * cols), trn_img)?.mapv(|x| (x as f64) / 255.),
        ndarray::Array::from_shape_vec((trn_size, 1), trn_lbl)?,
    );


    println!("whitening");
    // reduce to 50 dimension without whitening
    let ds = linfa_reduction::Pca::params(50)
        .whiten(false)
        .fit(&ds)
        .unwrap()
        .transform(ds);


    println!("calculating");
    // calculate a two-dimensional embedding with Barnes-Hut t-SNE
    let ds = linfa_tsne::TSneParams::embedding_size(2)
        .perplexity(50.0)
        .approx_threshold(0.5)
        .max_iter(1000)
        .transform(ds)?;

    // export data to dat
    export_data(ds);

    Ok(())
}

