use std::fs::{File, create_dir_all};
use std::io::{Write, BufReader, BufRead, BufWriter};
use std::error::Error;
use std::path::Path;
use std::f64::consts::PI;
use rand::Rng;
use serde::{Deserialize};

// -----------------------------------
// Constants and Configuration
// -----------------------------------
const NX: usize = 64;
const NY: usize = 64;
const NZ: usize = 8;   // 3rd spatial dimension
const NW: usize = 4;   // 5th dimension (compact)
const DX: f64 = 5e-4;
const DT: f64 = 5e-6;
const STEPS: usize = 500;
const OUTPUT_INTERVAL: usize = 50;

const NUM_FIELDS: usize = 9; 
// Fields indexing layout (per grid point):
// 0: photon distribution
// 1: axion distribution
// 2: neutrino distribution
// 3: energy density
// 4: metric perturbation
// 5: torsion_xx
// 6: torsion_xy
// 7: torsion_xz
// 8: chiral_odd_component (to test CP-violation / symmetry breaking)

// Coupling constants & parameters
const COUPLING_PH_AX: f64 = 1e-3;
const COUPLING_AX_NU: f64 = 1e-4;
const TORSION_STRENGTH: f64 = 1e-3;
const LIGHT_SPEED: f64 = 1.0; // dimensionless
const G_NEWTON: f64 = 6.7e-11; // dimensionful, scaled down
const RUNGE_KUTTA_STAGES: usize = 4;

// -----------------------------------
// Data Structures
// -----------------------------------

#[derive(Debug, Deserialize)]
struct DecayMode {
    products: String,
    branching_ratio: f64,
}

#[derive(Debug, Deserialize)]
struct Meson {
    name: String,
    mass: f64,
    spin: u8,
    quark_content: String,
    lifetime: f64,
    decay_modes: String,
}

// Unified field data structure for a single time step
// Stored as a 5D array flattened: dimension ordering: (W, Z, Y, X, field)
struct FieldData {
    data: Vec<f64>,
}

impl FieldData {
    fn new() -> Self {
        let size = NX * NY * NZ * NW * NUM_FIELDS;
        let mut data = vec![0.0; size];
        
        // Initialize fields with some physically motivated setup
        // Photon ~0.01, Axion ~0.005, Neutrino ~0.001, energy ~1.0, others ~0
        for w in 0..NW {
            for z in 0..NZ {
                for y in 0..NY {
                    for x in 0..NX {
                        let idx = Self::fidx(x,y,z,w,0);
                        data[idx + 0] = 0.01; // photon
                        data[idx + 1] = 0.005; // axion
                        data[idx + 2] = 0.001; // neutrino
                        data[idx + 3] = 1.0;   // energy
                        data[idx + 4] = 0.0;   // metric
                        data[idx + 5] = 0.0;   // torsion_xx
                        data[idx + 6] = 0.0;   // torsion_xy
                        data[idx + 7] = 0.0;   // torsion_xz
                        data[idx + 8] = 0.0;   // chiral_odd_component
                    }
                }
            }
        }

        FieldData { data }
    }

    #[inline]
    fn fidx(x:usize,y:usize,z:usize,w:usize,field_off:usize) -> usize {
        field_off + NUM_FIELDS*(x + NX*(y + NY*(z + NZ*w)))
    }

    fn get(&self, x:usize,y:usize,z:usize,w:usize, field:usize) -> f64 {
        self.data[Self::fidx(x,y,z,w,field)]
    }

    fn set(&mut self, x:usize,y:usize,z:usize,w:usize, field:usize, val:f64) {
        let idx = Self::fidx(x,y,z,w,field);
        self.data[idx] = val;
    }
}

// -----------------------------------
// External Data Loading: Mesons
// -----------------------------------

fn load_mesons(file_path: &str) -> Result<Vec<Meson>, Box<dyn Error>> {
    let mut rdr = csv::Reader::from_path(file_path)?;
    let mut mesons = Vec::new();
    for result in rdr.records() {
        let record = result?;
        let name = record.get(0).unwrap().to_string();
        let mass: f64 = record.get(1).unwrap().parse()?;
        let spin: u8 = record.get(2).unwrap().parse()?;
        let quark_content = record.get(3).unwrap().to_string();
        let lifetime: f64 = record.get(4).unwrap().parse()?;
        let decay_str = record.get(5).unwrap().to_string();
        mesons.push(Meson {
            name,
            mass,
            spin,
            quark_content,
            lifetime,
            decay_modes: decay_str,
        });
    }
    Ok(mesons)
}

fn parse_decay_modes(decay_str: &str) -> Vec<(Vec<String>, f64)> {
    // decay_str example: "mu+ νmu:99.987, e+ νe:0.013"
    let mut modes = Vec::new();
    for part in decay_str.split(',') {
        let p = part.trim();
        let parts: Vec<&str> = p.split(':').collect();
        if parts.len() == 2 {
            let products_str = parts[0].trim();
            let br_str = parts[1].trim();
            let br = br_str.parse::<f64>().unwrap_or(0.0);
            let products: Vec<String> = products_str.split('+')
                .map(|s| s.trim().to_string()).collect();
            modes.push((products, br));
        }
    }
    modes
}

// Simulate meson decay to produce neutrinos or other fields
fn simulate_decay(meson: &Meson) -> Option<Vec<String>> {
    let modes = parse_decay_modes(&meson.decay_modes);
    let mut rng = rand::thread_rng();
    let val: f64 = rng.gen_range(0.0..100.0);
    let mut cumulative = 0.0;
    for (prod, br) in &modes {
        cumulative += br;
        if val <= cumulative {
            return Some(prod.clone());
        }
    }
    None
}

// -----------------------------------
// Physics Kernels
// -----------------------------------

// Implement a measure-preserving transformation for photon-axion-neutrino fields
fn measure_preserving_transform(fields:&mut FieldData) {
    // For each point, apply balanced transformations:
    // photon -> axion -> neutrino -> photon loop
    // Adjust so that total integral is preserved.
    let size = NX*NY*NZ*NW;
    for idx_3d in 0..size {
        let base = idx_3d*NUM_FIELDS;
        let ph = fields.data[base+0];
        let ax = fields.data[base+1];
        let nu = fields.data[base+2];

        // photon to axion
        let delta_p_to_a = COUPLING_PH_AX * ph * DT;
        let ph_new = ph - delta_p_to_a;
        let ax_new = ax + delta_p_to_a;

        // axion to neutrino
        let delta_a_to_n = COUPLING_AX_NU * ax_new * DT;
        let ax_new2 = ax_new - delta_a_to_n;
        let nu_new = nu + delta_a_to_n;

        // neutrino back to photon (to close loop)
        let delta_n_to_p = COUPLING_AX_NU * nu_new * 0.5 * DT;
        let nu_final = nu_new - delta_n_to_p;
        let ph_final = ph_new + delta_n_to_p;

        fields.data[base+0] = ph_final;
        fields.data[base+1] = ax_new2;
        fields.data[base+2] = nu_final;
    }
}

// Higher-order finite difference approximations for spatial derivatives
// We'll implement a 4th-order accurate Laplacian in (X,Y,Z) and a finite difference in W as well.
fn periodic_idx(i: isize, max: usize) -> usize {
    let mut ii = i;
    while ii < 0 {
        ii += max as isize;
    }
    (ii as usize) % max
}

fn laplacian_4th(arr: &FieldData, field_id: usize) -> FieldData {
    // Compute 4th-order spatial Laplacian in X,Y,Z for each W slice
    let mut res = FieldData{data: arr.data.clone()};
    for w in 0..NW {
        for z in 0..NZ {
            for y in 0..NY {
                for x in 0..NX {
                    let xm2 = periodic_idx(x as isize - 2, NX);
                    let xm1 = periodic_idx(x as isize - 1, NX);
                    let xp1 = periodic_idx(x as isize + 1, NX);
                    let xp2 = periodic_idx(x as isize + 2, NX);

                    let ym2 = periodic_idx(y as isize - 2, NY);
                    let ym1 = periodic_idx(y as isize - 1, NY);
                    let yp1 = periodic_idx(y as isize + 1, NY);
                    let yp2 = periodic_idx(y as isize + 2, NY);

                    let zm2 = periodic_idx(z as isize - 2, NZ);
                    let zm1 = periodic_idx(z as isize - 1, NZ);
                    let zp1 = periodic_idx(z as isize + 1, NZ);
                    let zp2 = periodic_idx(z as isize + 2, NZ);

                    let c = arr.get(x,y,z,w,field_id);
                    let xm2v = arr.get(xm2,y,z,w,field_id);
                    let xm1v = arr.get(xm1,y,z,w,field_id);
                    let xp1v = arr.get(xp1,y,z,w,field_id);
                    let xp2v = arr.get(xp2,y,z,w,field_id);

                    let ym2v = arr.get(x,ym2,z,w,field_id);
                    let ym1v = arr.get(x,ym1,z,w,field_id);
                    let yp1v = arr.get(x,yp1,z,w,field_id);
                    let yp2v = arr.get(x,yp2,z,w,field_id);

                    let zm2v = arr.get(x,y,zm2,w,field_id);
                    let zm1v = arr.get(x,y,zm1,w,field_id);
                    let zp1v = arr.get(x,y,zp1,w,field_id);
                    let zp2v = arr.get(x,y,zp2,w,field_id);

                    let lap_x = (-xm2v + 16.0*xm1v - 30.0*c + 16.0*xp1v - xp2v) / (12.0*DX*DX);
                    let lap_y = (-ym2v + 16.0*ym1v - 30.0*c + 16.0*yp1v - yp2v) / (12.0*DX*DX);
                    let lap_z = (-zm2v + 16.0*zm1v - 30.0*c + 16.0*zp1v - zp2v) / (12.0*DX*DX);

                    let val = lap_x + lap_y + lap_z;
                    res.set(x,y,z,w, field_id, val);
                }
            }
        }
    }
    res
}

// Simple finite difference in W-dimension (compact dimension)
fn derivative_w(arr: &FieldData, field_id: usize) -> FieldData {
    let mut res = FieldData{data: arr.data.clone()};
    for w in 0..NW {
        let wp = (w+1)%NW;
        let wm = if w == 0 { NW-1 } else { w-1 };
        for z in 0..NZ {
            for y in 0..NY {
                for x in 0..NX {
                    let cp = arr.get(x,y,z,wp,field_id);
                    let cm = arr.get(x,y,z,wm,field_id);
                    let dw = (cp - cm)/(2.0*DX); // same DX scale for simplicity
                    res.set(x,y,z,w, field_id, dw);
                }
            }
        }
    }
    res
}

// Boltzmann update includes collision integrals and momentum dependence (conceptual)
fn boltzmann_update_momentum(/* parameters */) {
    // This would integrate over momentum space, updating f_neutrino and f_axion distributions.
    // Complexity omitted due to length. Here you'd implement a momentum grid and collision integrals.
}

// Runge-Kutta time stepping for PDE updates
fn time_step_rk4(fields: &mut FieldData) {
    // Example: apply laplacian diffusion to photon field as a test
    // Similarly integrate metric/torsion evolution. 
    // Realistically, you'd have a system of PDEs and solve them simultaneously.
    
    // For demonstration: photon field diffusion
    let photon_id = 0;
    // We'll do photon diffusion as a stand-in PDE:
    // d(photon)/dt = D * Laplacian(photon)
    let D = 1e-3;

    // k1
    let lap_p = laplacian_4th(fields, photon_id);
    let mut k1 = FieldData{data:vec![0.0;fields.data.len()]};
    for i in 0..fields.data.len()/(NUM_FIELDS) {
        let base = i*NUM_FIELDS;
        let p = fields.data[base+photon_id];
        let lp = lap_p.data[base+photon_id];
        k1.data[base+photon_id] = D * lp;
    }

    // apply k1/2 step
    let mut f_temp = FieldData{data:fields.data.clone()};
    for i in 0..fields.data.len()/(NUM_FIELDS) {
        let base = i*NUM_FIELDS;
        f_temp.data[base+photon_id] += k1.data[base+photon_id]*(DT/2.0);
    }

    // k2
    let lap_p2 = laplacian_4th(&f_temp, photon_id);
    let mut k2 = FieldData{data:vec![0.0;fields.data.len()]};
    for i in 0..fields.data.len()/(NUM_FIELDS) {
        let base = i*NUM_FIELDS;
        let lp2 = lap_p2.data[base+photon_id];
        k2.data[base+photon_id] = D*lp2;
    }

    // apply k2/2
    let mut f_temp2 = FieldData{data:fields.data.clone()};
    for i in 0..fields.data.len()/(NUM_FIELDS) {
        let base = i*NUM_FIELDS;
        f_temp2.data[base+photon_id] += k2.data[base+photon_id]*(DT/2.0);
    }

    // k3
    let lap_p3 = laplacian_4th(&f_temp2, photon_id);
    let mut k3 = FieldData{data:vec![0.0;fields.data.len()]};
    for i in 0..fields.data.len()/(NUM_FIELDS) {
        let base = i*NUM_FIELDS;
        let lp3 = lap_p3.data[base+photon_id];
        k3.data[base+photon_id] = D*lp3;
    }

    // apply k3 fully
    let mut f_temp3 = FieldData{data:fields.data.clone()};
    for i in 0..fields.data.len()/(NUM_FIELDS) {
        let base = i*NUM_FIELDS;
        f_temp3.data[base+photon_id] += k3.data[base+photon_id]*DT;
    }

    // k4
    let lap_p4 = laplacian_4th(&f_temp3, photon_id);
    let mut k4 = FieldData{data:vec![0.0;fields.data.len()]};
    for i in 0..fields.data.len()/(NUM_FIELDS) {
        let base = i*NUM_FIELDS;
        let lp4 = lap_p4.data[base+photon_id];
        k4.data[base+photon_id] = D*lp4;
    }

    // combine
    for i in 0..fields.data.len()/(NUM_FIELDS) {
        let base = i*NUM_FIELDS;
        fields.data[base+photon_id] += (DT/6.0)*(k1.data[base+photon_id] + 2.0*k2.data[base+photon_id] +
                                                 2.0*k3.data[base+photon_id] + k4.data[base+photon_id]);
    }

    // Similar steps would be applied to axion, neutrino, and metric/torsion fields with their respective PDEs.
    // Omitted for brevity.
}

fn compute_averages(fields: &FieldData) -> (f64,f64,f64,f64) {
    let size_4d = NX*NY*NZ*NW;
    let mut sum_ph=0.0; let mut sum_ax=0.0; let mut sum_nu=0.0; let mut sum_e=0.0;
    for i in 0..size_4d {
        let base = i*NUM_FIELDS;
        sum_ph += fields.data[base+0];
        sum_ax += fields.data[base+1];
        sum_nu += fields.data[base+2];
        sum_e += fields.data[base+3];
    }
    let norm = size_4d as f64;
    (sum_ph/norm, sum_ax/norm, sum_nu/norm, sum_e/norm)
}

// Update metric and torsion fields based on neutrino spin density
fn update_ec_torsion_metric(fields: &mut FieldData) {
    let size_4d = NX*NY*NZ*NW;
    for i in 0..size_4d {
        let base = i*NUM_FIELDS;
        let nu = fields.data[base+2];
        // Simple coupling: torsion_xx ~ torsion_xx + TORSION_STRENGTH * nu * DT
        fields.data[base+5] += TORSION_STRENGTH * nu * DT * 1e-6; // torsion_xx
        fields.data[base+6] += TORSION_STRENGTH * nu * DT * 1e-6; // torsion_xy
        // update metric:
        fields.data[base+4] += (fields.data[base+5] + fields.data[base+6]) * DT * 1e-2;
    }
}

fn write_snapshot(step: usize, fields:&FieldData) -> Result<(),Box<dyn Error>> {
    let snapshot_dir = "snapshots";
    if !Path::new(snapshot_dir).exists() {
        create_dir_all(snapshot_dir)?;
    }
    let fname = format!("{}/fields_{:05}.bin", snapshot_dir, step);
    let mut f = BufWriter::new(File::create(fname)?);
    for val in fields.data.iter() {
        f.write_all(&val.to_le_bytes())?;
    }
    Ok(())
}

fn main() -> Result<(),Box<dyn Error>> {
    // Attempt to load meson data for realistic decay processes
    // (If file not found, we just proceed)
    let mesons = load_mesons("mesons.csv").unwrap_or_else(|_| Vec::new());

    let mut fields = FieldData::new();
    let mut scale_factor = 1.0;

    let mut file = BufWriter::new(File::create("results.csv")?);
    writeln!(file,"time,a,avg_photon,avg_axion,avg_neutrino,avg_energy")?;

    for step in 0..STEPS {
        // 1. Measure-preserving transformation among photon, axion, neutrino
        measure_preserving_transform(&mut fields);

        // 2. Einstein-Cartan torsion/metric update
        update_ec_torsion_metric(&mut fields);

        // 3. Solve PDEs for distributions using RK4
        time_step_rk4(&mut fields);

        // Could also integrate meson decays:
        // For demonstration, randomly pick a meson and simulate a decay:
        if !mesons.is_empty() {
            let mut rng = rand::thread_rng();
            let m_idx = rng.gen_range(0..mesons.len());
            if let Some(prods) = simulate_decay(&mesons[m_idx]) {
                // Suppose meson decays produce neutrinos: increase neutrino field slightly
                // In a real scenario, map decay products to field increments
                let size_4d = NX*NY*NZ*NW;
                // Add neutrinos uniformly as a crude approximation
                for i in 0..size_4d {
                    let base = i*NUM_FIELDS;
                    for p in &prods {
                        match p.as_str() {
                            "νμ" | "νe" | "ντ" => {
                                fields.data[base+2] += 1e-6;
                            },
                            "γ" => {
                                fields.data[base+0] += 1e-6;
                            },
                            _ => {}
                        }
                    }
                }
            }
        }

        // Compute averages
        let (avg_ph, avg_ax, avg_nu, avg_e) = compute_averages(&fields);

        // Update scale factor (toy Friedmann)
        scale_factor += DT * scale_factor * avg_e.sqrt();

        let time = step as f64 * DT;
        writeln!(file, "{},{},{},{},{},{}", time, scale_factor, avg_ph, avg_ax, avg_nu, avg_e)?;

        if step % OUTPUT_INTERVAL == 0 {
            println!("Step {}: a={:.5}, ph={:.5}, ax={:.5}, nu={:.5}, E={:.5}",
                     step, scale_factor, avg_ph, avg_ax, avg_nu, avg_e);
            write_snapshot(step,&fields)?;
        }
    }

    println!("Simulation complete. Results in results.csv and snapshots/");

    Ok(())
}
