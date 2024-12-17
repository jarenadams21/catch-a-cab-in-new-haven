use std::fs::File;
use std::io::Write;
use std::f64::consts::PI;

/// We now use natural units: c = ħ = k_B = 1, G set so that M_pl ~ 1 in chosen units.
/// For demonstration, we set dimensionless scales. In real codes, carefully define these.
///
/// Let's define:
/// - Energy density scale: We'll treat the initial energy density as O(1) dimensionless.
/// - Distances: DX chosen arbitrarily small.
/// - Time steps: DT chosen small accordingly.
///
/// We assume a simple radiation-like EoS: p = epsilon/3.
/// You can replace this with a more complex EoS from QCD tables if desired.

/// Grid parameters
const NX: usize = 100;
const NY: usize = 100;
const DX: f64 = 1e-3;   // dimensionless spatial step
const DT: f64 = 1e-5;   // dimensionless time step
const STEPS: usize = 1000;  // more steps to see some evolution
const OUTPUT_INTERVAL: usize = 100; // Write snapshots every 100 steps

/// Simple EoS for demonstration:
fn eos_pressure(eps: f64) -> f64 {
    // Radiation-like fluid: p = eps/3
    eps / 3.0
}

/// A toy metric perturbation function:
/// We'll just store a small fluctuation in the metric as an extra field.
/// In a real scenario, you'd solve Einstein's equations or have a known metric background.
fn metric_perturbation_update(metric: &mut Vec<f64>) {
    // Just let metric relax slightly each step:
    for val in metric.iter_mut() {
        *val *= 0.999999; 
    }
}

/// Laplacian on a 2D torus (periodic boundary)
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

/// PDE update
fn update_step(
    photon: &mut Vec<f64>,
    neutrino: &mut Vec<f64>,
    energy: &mut Vec<f64>,
    a_field: &mut Vec<f64>,
    h_field: &mut Vec<f64>,
    metric: &mut Vec<f64>,
) {
    let size = NX*NY;
    let mut new_ph = vec![0.0; size];
    let mut new_nu = vec![0.0; size];
    let mut new_e = vec![0.0; size];

    // Mass terms or couplings for scalar fields (toy):
    // In a more realistic scenario, these would come from a chosen potential or model.
    let ma2 = 1e-4; // ALP mass^2 (dimensionless)
    let mh2 = 1e-3; // Higgs-like scalar mass^2 (dimensionless)

    for y in 0..NY {
        for x in 0..NX {
            let idx = x + NX*y;
            let e_val = energy[idx];
            let _p_val = eos_pressure(e_val);

            // Diffusion-like updates for photon and neutrino densities:
            let dph_dt = 1e-3 * lap(photon, x, y);
            let dnu_dt = 1e-3 * lap(neutrino, x, y);

            let a_val = a_field[idx];
            let h_val = h_field[idx];

            // Energy equation update:
            // Include a tiny feedback from scalar fields to energy:
            let de_dt = 1e-4 * lap(energy, x, y) - 1e-5 * (ma2 * a_val + mh2 * h_val * h_val);

            new_ph[idx] = photon[idx] + DT*dph_dt;
            new_nu[idx] = neutrino[idx] + DT*dnu_dt;
            new_e[idx] = energy[idx] + DT*de_dt;
        }
    }

    photon.copy_from_slice(&new_ph);
    neutrino.copy_from_slice(&new_nu);
    energy.copy_from_slice(&new_e);

    // Update scalar fields (toy damp)
    for idx in 0..size {
        a_field[idx] *= 1.0 - 1e-6;
        h_field[idx] *= 1.0 - 1e-6;
    }

    // Update metric perturbation field
    metric_perturbation_update(metric);
}

fn main() {
    let size = NX*NY;

    // Initialize fields with more moderate dimensionless values
    let mut photon_density = vec![0.1; size];   // dimensionless photon number density
    let mut neutrino_density = vec![0.01; size]; // dimensionless neutrino number density
    let mut energy_density = vec![1.0; size];   // dimensionless energy density ~1

    // ALP and Higgs-like fields:
    let mut a_field = vec![1e-3; size];  
    let mut h_field = vec![1e-3; size];  

    // Metric perturbation field:
    let mut metric_field = vec![1e-4; size];  // small perturbation

    let mut file = File::create("results.csv").unwrap();
    writeln!(file,"time,avg_photon_density,avg_neutrino_density,avg_energy_density,avg_metric_pert").unwrap();

    // We will also write snapshots at intervals:
    std::fs::create_dir_all("snapshots").unwrap();

    for step in 0..STEPS {
        let t = step as f64 * DT;

        update_step(
            &mut photon_density,
            &mut neutrino_density,
            &mut energy_density,
            &mut a_field,
            &mut h_field,
            &mut metric_field,
        );

        let vol = (NX*NY) as f64;
        let avg_ph = photon_density.iter().sum::<f64>()/vol;
        let avg_nu = neutrino_density.iter().sum::<f64>()/vol;
        let avg_e = energy_density.iter().sum::<f64>()/vol;
        let avg_m = metric_field.iter().sum::<f64>()/vol;

        writeln!(file,"{},{},{},{},{}",t,avg_ph,avg_nu,avg_e,avg_m).unwrap();

        // Write snapshots every OUTPUT_INTERVAL steps
        if step % OUTPUT_INTERVAL == 0 {
            let fname = format!("snapshots/fields_{:05}.bin", step);
            let mut f = File::create(fname).unwrap();
            // Write out photon, neutrino, energy, a_field, h_field, metric in order
            // Each field: size * f64
            for val in &photon_density { f.write_all(&val.to_le_bytes()).unwrap(); }
            for val in &neutrino_density { f.write_all(&val.to_le_bytes()).unwrap(); }
            for val in &energy_density { f.write_all(&val.to_le_bytes()).unwrap(); }
            for val in &a_field { f.write_all(&val.to_le_bytes()).unwrap(); }
            for val in &h_field { f.write_all(&val.to_le_bytes()).unwrap(); }
            for val in &metric_field { f.write_all(&val.to_le_bytes()).unwrap(); }
        }
    }

    println!("Simulation complete. Data saved to snapshots/ and results.csv");
}
