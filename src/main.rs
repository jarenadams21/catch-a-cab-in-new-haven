//! DISCLAIMER:
//! This code remains a *highly simplified demonstration* and is not a full-fledged simulation of 
//! lattice QCD, FLRW cosmology, gravitational waves, or Boltzmann transport. It provides a conceptual 
//! template for how one might incorporate these elements into a codebase.
//! 
//! For actual research applications, the following steps would be necessary:
//! - Use real lattice QCD EoS data tables from collaborations such as HotQCD, rather than a toy parameterization.
//! - Employ more sophisticated PDE solvers (e.g., high-order methods, adaptive mesh refinement).
//! - Incorporate momentum-dependent distribution functions and collision integrals for neutrinos and 
//!   axions when solving the Boltzmann equation, rather than the simplistic placeholder shown here.
//! - Solve Einstein’s field equations or use known cosmological backgrounds for FLRW + perturbations 
//!   if you want accurate metric and gravitational wave evolution.
//! - Validate against known benchmarks, and ensure consistent unit conversions if using mixed units.
//!
//! The purpose of this code is purely illustrative, showing how fields and parameters might be structured, 
//! not how to achieve physically robust results.

// To compile and run:
// 1. Ensure you have Rust and cargo installed.
// 2. `cargo build --release`
// 3. `./target/release/your_executable_name`
// 4. The code will output snapshots and a CSV file for visualization.

use std::fs::File;
use std::io::Write;
use std::f64::consts::PI;

const NX: usize = 100;
const NY: usize = 100;
const DX: f64 = 1e-3;   
const DT: f64 = 1e-5;   
const STEPS: usize = 2000;
const OUTPUT_INTERVAL: usize = 100;

// Toy parameters for EoS inspired by lattice QCD (NOT REAL DATA)
const EPS_TRANS: f64 = 0.7;
const DELTA: f64 = 0.1;
const P_LOW: f64 = 0.1; // Fractional for the low-density regime

/// Toy equation of state function.
/// In actual research, replace this with real lattice QCD data from HotQCD or other collaborations.
/// Typically, you'd read a table of (epsilon, p) and interpolate.
fn eos_pressure(eps: f64) -> f64 {
    let x = (eps - EPS_TRANS)/DELTA;
    let w_qgp = 0.5*(1.0 + x.tanh());
    let p_qgp = (1.0/3.0)*eps;
    let p_hg = P_LOW*eps;
    w_qgp*p_qgp + (1.0 - w_qgp)*p_hg
}

/// Update the scale factor a(t) using a simplified Friedmann equation.
/// Realistically, you'd solve full Friedmann equations with correct units and gravitational constants.
/// For demonstration only.
fn update_scale_factor(a: f64, eps: f64) -> f64 {
    a + DT * a * eps.abs().sqrt()
}

/// Compute a discrete Laplacian with periodic boundary conditions on a 2D torus.
/// In a realistic scenario, use higher-order finite differences or spectral methods.
fn lap(arr: &Vec<f64>, x: usize, y: usize) -> f64 {
    let idx = x + NX*y;
    let xm = if x == 0 {NX-1} else {x-1};
    let xp = if x == NX-1 {0} else {x+1};
    let ym = if y == 0 {NY-1} else {y-1};
    let yp = if y == NY-1 {0} else {y+1};
    let c = arr[idx];
    let neighbors = arr[xm+NX*y] + arr[xp+NX*y] + arr[x+NX*ym] + arr[x+NX*yp];
    (neighbors - 4.0*c)/(DX*DX)
}

/// Placeholder Boltzmann update routine.
/// Real Boltzmann solvers involve momentum-dependent distributions and collision integrals.
/// This is just a toy damping factor as a placeholder.
fn boltzmann_update(neutrino: &mut Vec<f64>, a_field: &mut Vec<f64>) {
    let size = NX*NY;
    for i in 0..size {
        neutrino[i] *= 1.0 - 1e-7 * a_field[i].sqrt();
    }
}

/// Toy gravitational wave (GW) update.
/// Real GW evolution requires solving tensor mode equations in an FLRW background.
fn gw_update(gw: &mut Vec<f64>, a_new: f64, a_old: f64) {
    let redshift_factor = a_old/a_new;
    for val in gw.iter_mut() {
        *val *= redshift_factor;
        *val *= 0.999999;
    }
}

/// PDE update step (still toy-level).
/// Does not handle the full complexity of a realistic cosmological PDE solver.
fn update_step(
    photon: &mut Vec<f64>,
    neutrino: &mut Vec<f64>,
    energy: &mut Vec<f64>,
    a_field: &mut Vec<f64>,
    h_field: &mut Vec<f64>,
    metric: &mut Vec<f64>,
    gw_field: &mut Vec<f64>,
    a: f64,
) {
    let size = NX*NY;
    let mut new_ph = vec![0.0; size];
    let mut new_nu = vec![0.0; size];
    let mut new_e = vec![0.0; size];

    let ma2 = 1e-4; // toy ALP mass^2
    let mh2 = 1e-3; // toy Higgs-like mass^2

    for y in 0..NY {
        for x in 0..NX {
            let idx = x + NX*y;
            // We do not actually factor in scale factor into the PDE currently.
            // A full simulation would use physical derivatives and scale factor expansions.
            let dph_dt = 1e-3 * lap(photon, x, y);
            let dnu_dt = 1e-3 * lap(neutrino, x, y);

            let a_val = a_field[idx];
            let h_val = h_field[idx];
            let de_dt = 1e-4 * lap(energy, x, y) - 1e-5*(ma2*a_val + mh2*h_val*h_val);

            new_ph[idx] = photon[idx] + DT*dph_dt;
            new_nu[idx] = neutrino[idx] + DT*dnu_dt;
            new_e[idx] = energy[idx] + DT*de_dt;
        }
    }

    photon.copy_from_slice(&new_ph);
    neutrino.copy_from_slice(&new_nu);
    energy.copy_from_slice(&new_e);

    // Boltzmann placeholder
    boltzmann_update(neutrino, a_field);

    // Dampen scalar fields slightly
    for idx in 0..size {
        a_field[idx] *= 1.0 - 1e-6;
        h_field[idx] *= 1.0 - 1e-6;
    }

    // Dampen metric perturbations slightly
    for val in metric.iter_mut() {
        *val *= 0.999999;
    }
}

fn main() {
    let size = NX*NY;

    // Initial conditions (toy values)
    let mut photon_density = vec![0.1; size];
    let mut neutrino_density = vec![0.01; size];
    let mut energy_density = vec![1.0; size];
    let mut a_field = vec![1e-3; size];
    let mut h_field = vec![1e-3; size];
    let mut metric_field = vec![1e-4; size];
    let mut gw_field = vec![1e-6; size];

    let mut a = 1.0; // scale factor
    let mut file = File::create("results.csv").unwrap();
    writeln!(
        file,
        "time,a,avg_photon_density,avg_neutrino_density,avg_energy_density,avg_metric_pert,avg_gw"
    ).unwrap();

    std::fs::create_dir_all("snapshots").unwrap();

    for step in 0..STEPS {
        let t = step as f64 * DT;
        let vol = (NX*NY) as f64;

        // Update scale factor
        let avg_e = energy_density.iter().sum::<f64>()/vol;
        let a_old = a;
        a = update_scale_factor(a, avg_e);

        // PDE update
        update_step(
            &mut photon_density,
            &mut neutrino_density,
            &mut energy_density,
            &mut a_field,
            &mut h_field,
            &mut metric_field,
            &mut gw_field,
            a,
        );

        // GW update after scale factor changes
        gw_update(&mut gw_field, a, a_old);

        let avg_ph = photon_density.iter().sum::<f64>()/vol;
        let avg_nu = neutrino_density.iter().sum::<f64>()/vol;
        let avg_e = energy_density.iter().sum::<f64>()/vol;
        let avg_m = metric_field.iter().sum::<f64>()/vol;
        let avg_gw = gw_field.iter().sum::<f64>()/vol;

        writeln!(file,"{},{},{},{},{},{},{}",t,a,avg_ph,avg_nu,avg_e,avg_m,avg_gw).unwrap();

        if step % OUTPUT_INTERVAL == 0 {
            let fname = format!("snapshots/fields_{:05}.bin", step);
            let mut f = File::create(fname).unwrap();
            // Save all fields for post-processing
            for val in &photon_density { f.write_all(&val.to_le_bytes()).unwrap(); }
            for val in &neutrino_density { f.write_all(&val.to_le_bytes()).unwrap(); }
            for val in &energy_density { f.write_all(&val.to_le_bytes()).unwrap(); }
            for val in &a_field { f.write_all(&val.to_le_bytes()).unwrap(); }
            for val in &h_field { f.write_all(&val.to_le_bytes()).unwrap(); }
            for val in &metric_field { f.write_all(&val.to_le_bytes()).unwrap(); }
            for val in &gw_field { f.write_all(&val.to_le_bytes()).unwrap(); }
        }
    }

    println!("Simulation complete. Data saved to snapshots/ and results.csv");
}
