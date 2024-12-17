use std::fs::File;
use std::io::Write;

/// Physical constants
const C: f64 = 3.0e8;
const HBAR: f64 = 1.054571817e-34;
const G: f64 = 6.67430e-11;
const K_B: f64 = 1.380649e-23;
const MP: f64 = 2.435e18; // reduced Planck mass in GeV (~2.435e18)
const F: f64 = 1e-32;      // Tiny gauge coupling controlling cosmological constant
const F_TILDE: f64 = 1e-10; // Coupling for ALP mass
const XI_H: f64 = 1e-2;
const LAMBDA_H: f64 = 1e-1; // Higgs self-coupling

/// Lattice/grid size
const NX: usize = 100;
const NY: usize = 100;
const DX: f64 = 1.0e-15;
const DT: f64 = 5.0e-44; // A very small timestep, just for illustration
const STEPS: usize = 200;

/// Equation of state pressure function
fn eos_pressure(eps: f64) -> f64 {
    let eps_crit = 1.0e35;
    let delta = 0.2e35;
    let w_qgp = 0.5*(1.0+((eps - eps_crit)/delta).tanh());
    let p_qgp = (1.0/3.0)*eps;
    let p_hg = 0.15*eps;
    w_qgp*p_qgp+(1.0 - w_qgp)*p_hg
}

/// Scalar mass corrections from gravity:
fn alp_mass_grav() -> f64 {
    F_TILDE*MP/(4.0*(3.0_f64).sqrt())
}

fn higgs_mass_grav_sq() -> f64 {
    (F*F*XI_H*MP*MP)/4.0
}

/// Update step performing a PDE iteration
fn update_step(
    photon: &mut Vec<f64>,
    neutrino: &mut Vec<f64>,
    energy: &mut Vec<f64>,
    a_field: &mut Vec<f64>,
    h_field: &mut Vec<f64>
) {
    let size = NX*NY;
    let mut new_ph = vec![0.0; size];
    let mut new_nu = vec![0.0; size];
    let mut new_e = vec![0.0; size];

    let lap = |arr: &Vec<f64>, x:usize, y:usize| {
        let idx = x + NX*y;
        let xm = if x==0 {NX-1} else {x-1};
        let xp = if x==NX-1 {0} else {x+1};
        let ym = if y==0 {NY-1} else {y-1};
        let yp = if y==NY-1 {0} else {y+1};
        let c = arr[idx];
        let neighbors = arr[xm+NX*y]+arr[xp+NX*y]+arr[x+NX*ym]+arr[x+NX*yp];
        (neighbors-4.0*c)/(DX*DX)
    };

    for y in 0..NY {
        for x in 0..NX {
            let idx = x+NX*y;
            let e_val = energy[idx];
            let _p_val = eos_pressure(e_val);
            let dph_dt = 1e-3*lap(photon,x,y);
            let dnu_dt = 1e-3*lap(neutrino,x,y);

            let a_val = a_field[idx];
            let h_val = h_field[idx];
            let ma = alp_mass_grav();
            let mh2 = higgs_mass_grav_sq();

            // Include minimal energy feedback
            let de_dt = 1e-4*lap(energy,x,y) - 1e-5*(ma*ma*a_val + mh2*h_val*h_val);

            new_ph[idx] = photon[idx]+DT*dph_dt;
            new_nu[idx] = neutrino[idx]+DT*dnu_dt;
            new_e[idx] = energy[idx]+DT*de_dt;
        }
    }

    // Overwrite old data with new updates
    photon.copy_from_slice(&new_ph);
    neutrino.copy_from_slice(&new_nu);
    energy.copy_from_slice(&new_e);

    // Slowly vary the scalar fields
    for idx in 0..size {
        a_field[idx] *= 1.0 - 1e-6;
        h_field[idx] *= 1.0 - 1e-6;
    }
}

fn main() {
    let size = NX*NY;
    let mut photon_density = vec![1e1; size];   // initial photon density
    let mut neutrino_density = vec![1e-20; size]; // initial neutrino density
    let mut energy_density = vec![3.2e35; size];   // high energy QGP-like
    let mut a_field = vec![1e-5; size];  // small ALP background
    let mut h_field = vec![1e-5; size];  // small Higgs background

    let mut file = File::create("results.csv").unwrap();
    writeln!(file,"time(s),avg_photon_density,avg_neutrino_density,avg_energy_density").unwrap();

    for step in 0..STEPS {
        let t = step as f64*DT;

        update_step(
            &mut photon_density,
            &mut neutrino_density,
            &mut energy_density,
            &mut a_field,
            &mut h_field
        );

        let vol = (NX*NY) as f64;
        let avg_ph = photon_density.iter().sum::<f64>()/vol;
        let avg_nu = neutrino_density.iter().sum::<f64>()/vol;
        let avg_e = energy_density.iter().sum::<f64>()/vol;

        writeln!(file,"{},{},{},{}",t,avg_ph,avg_nu,avg_e).unwrap();
    }

    // Write final data
    {
        let mut f_ph = File::create("photon_density_final.npy").unwrap();
        let mut f_nu = File::create("neutrino_density_final.npy").unwrap();
        let mut f_e = File::create("energy_density_final.npy").unwrap();

        for val in &photon_density {
            f_ph.write_all(&val.to_le_bytes()).unwrap();
        }
        for val in &neutrino_density {
            f_nu.write_all(&val.to_le_bytes()).unwrap();
        }
        for val in &energy_density {
            f_e.write_all(&val.to_le_bytes()).unwrap();
        }
    }

    println!("Simulation complete. Data saved to .npy and results.csv");
}
