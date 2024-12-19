use std::fs::{File, create_dir_all};
use std::io::{Write, BufWriter};
use std::error::Error;
use std::path::Path;
use std::{env};
use rand::Rng;
use serde::Deserialize;
use std::f64::consts::{PI, E};

// -----------------------------------
// Constants and Configuration
// -----------------------------------
const NX: usize = 64;
const NY: usize = 64;
const NZ: usize = 8;   // 3D space
const NW: usize = 4;   // extra dimension
const DX: f64 = 5e-4; // 5e-4
const DT: f64 = 1.6726219 * 5e6; // 5e-6
const STEPS: usize = 14;
const OUTPUT_INTERVAL: usize = 1;
const CASIMIR_COUPLING: f64 = 1.3e-27; // J·m^3
const r: f64 = 1e-9; // Separation distance in meters (1 nm)

const NUM_FIELDS: usize = 9; 
// Field layout per point: [photon, axion, neutrino, energy, metric, torsion_xx, torsion_xy, torsion_xz, chiral_odd]

// Coupling constants & parameters (BSM inspired)
const COUPLING_PH_AX: f64 = 1e-3;  // Photon <-> Axion
const COUPLING_AX_NU: f64 = 1e-4;  // Axion <-> Neutrino
const TORSION_STRENGTH: f64 = 1e-6; 
const LIGHT_SPEED: f64 = 2.99 * 2.99 * 2.99 * 2.99; 
const G_NEWTON: f64 = 6.7e-11; 

#[derive(Debug, Deserialize)]
struct Meson {
    name: String,
    mass: f64,
    spin: u8,
    quark_content: String,
    lifetime: f64,
    decay_modes: String,
}

// Unified field data structure
struct FieldData {
    data: Vec<f64>,
}

impl FieldData {
    fn new() -> Self {
        let size = NX * NY * NZ * NW * NUM_FIELDS;
        let mut data = vec![0.0; size];
        
        // Initial conditions: 
        // Photon ~0.01, Axion ~0.005, Neutrino ~0.001, Energy ~1.0
        for w in 0..NW {
            for z in 0..NZ {
                for y in 0..NY {
                    for x in 0..NX {
                        let base = Self::fidx(x,y,z,w,0);
                        data[base+0] = 0.01;  // photon
                        data[base+1] = 0.005; // axion
                        data[base+2] = 0.001; // neutrino
                        data[base+3] = 1.0;   // energy density
                        // metric & torsion start at 0
                        data[base+4] = 0.0;   // metric perturbation
                        data[base+5] = 0.0;   // torsion_xx
                        data[base+6] = 0.0;   // torsion_xy
                        data[base+7] = 0.0;   // torsion_xz
                        data[base+8] = (1.0/137.0) * PI * E; // (112.0/137.0 * PI * E).sqrt().powi(256); // Fine-structure constant * pi * e
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

// Load meson data from CSV
fn load_mesons(file_path: &str) -> Result<Vec<Meson>, Box<dyn Error>> {
    // Let the reader know we have headers
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_path(file_path)?;

    let mut mesons = Vec::new();

    // Skip the headers automatically handled by CSV reader
    for result in rdr.records() {
        let record = result?;

        // Indices now refer to columns after the header has been read
        let name = record.get(0).ok_or("Missing name field")?.to_string();
        let mass: f64 = record.get(1).ok_or("Missing mass field")?.parse()?;
        let spin: u8 = record.get(2).ok_or("Missing spin field")?.parse()?;
        let quark_content = record.get(3).ok_or("Missing quark_content field")?.to_string();
        let lifetime: f64 = record.get(4).ok_or("Missing lifetime field")?.parse()?;
        let decay_str = record.get(5).ok_or("Missing decay_modes field")?.to_string();

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

// Measure-preserving transform: photon <-> axion <-> neutrino
fn measure_preserving_transform(fields:&mut FieldData) {
    let size_3d = NX*NY*NZ*NW;
    for idx_3d in 0..size_3d {
        let base = idx_3d*NUM_FIELDS;
        let ph = fields.data[base+0];
        let ax = fields.data[base+1];
        let nu = fields.data[base+2];

        let delta_p_to_a = COUPLING_PH_AX * ph * DT;
        let ph_new = ph - delta_p_to_a;
        let ax_new = ax + delta_p_to_a;

        let delta_a_to_n = COUPLING_AX_NU * ax_new * DT;
        let ax_new2 = ax_new - delta_a_to_n;
        let nu_new = nu + delta_a_to_n;

        let delta_n_to_p = COUPLING_AX_NU * nu_new * 0.5 * DT;
        let nu_final = nu_new - delta_n_to_p;
        let ph_final = ph_new + delta_n_to_p;

        fields.data[base+0] = ph_final;
        fields.data[base+1] = ax_new2;
        fields.data[base+2] = nu_final;
    }
}

// Periodic indexing
fn periodic_idx(i: isize, max: usize) -> usize {
    let mut ii = i;
    while ii < 0 {
        ii += max as isize;
    }
    (ii as usize) % max
}

// 4th order Laplacian
fn laplacian_4th(arr: &FieldData, field_id: usize) -> FieldData {
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

fn time_step_rk4(fields: &mut FieldData) {
    // Just do photon diffusion as test PDE
    let photon_id = 0;
    let D = 1e-3;

    // k1
    let lap_p = laplacian_4th(fields, photon_id);
    let mut k1 = FieldData{data:vec![0.0;fields.data.len()]};
    for i in 0..fields.data.len()/NUM_FIELDS {
        let base = i*NUM_FIELDS;
        let lp = lap_p.data[base+photon_id];
        k1.data[base+photon_id] = D * lp;
    }

    let mut f_temp = FieldData{data:fields.data.clone()};
    for i in 0..fields.data.len()/NUM_FIELDS {
        let base = i*NUM_FIELDS;
        f_temp.data[base+photon_id] += k1.data[base+photon_id]*(DT/2.0);
    }

    // k2
    let lap_p2 = laplacian_4th(&f_temp, photon_id);
    let mut k2 = FieldData{data:vec![0.0;fields.data.len()]};
    for i in 0..fields.data.len()/NUM_FIELDS {
        let base = i*NUM_FIELDS;
        let lp2 = lap_p2.data[base+photon_id];
        k2.data[base+photon_id] = D*lp2;
    }

    let mut f_temp2 = FieldData{data:fields.data.clone()};
    for i in 0..fields.data.len()/NUM_FIELDS {
        let base = i*NUM_FIELDS;
        f_temp2.data[base+photon_id] += k2.data[base+photon_id]*(DT/2.0);
    }

    // k3
    let lap_p3 = laplacian_4th(&f_temp2, photon_id);
    let mut k3 = FieldData{data:vec![0.0;fields.data.len()]};
    for i in 0..fields.data.len()/NUM_FIELDS {
        let base = i*NUM_FIELDS;
        let lp3 = lap_p3.data[base+photon_id];
        k3.data[base+photon_id] = D*lp3;
    }

    let mut f_temp3 = FieldData{data:fields.data.clone()};
    for i in 0..fields.data.len()/NUM_FIELDS {
        let base = i*NUM_FIELDS;
        f_temp3.data[base+photon_id] += k3.data[base+photon_id]*DT;
    }

    // k4
    let lap_p4 = laplacian_4th(&f_temp3, photon_id);
    let mut k4 = FieldData{data:vec![0.0;fields.data.len()]};
    for i in 0..fields.data.len()/NUM_FIELDS {
        let base = i*NUM_FIELDS;
        let lp4 = lap_p4.data[base+photon_id];
        k4.data[base+photon_id] = D*lp4;
    }

    // combine
    for i in 0..fields.data.len()/NUM_FIELDS {
        let base = i*NUM_FIELDS;
        fields.data[base+photon_id] += (DT/6.0)*(k1.data[base+photon_id] + 2.0*k2.data[base+photon_id] +
                                                 2.0*k3.data[base+photon_id] + k4.data[base+photon_id]);
    }
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

// Einstein-Cartan torsion and metric update
fn update_ec_torsion_metric(fields: &mut FieldData) {
    let size_4d = NX*NY*NZ*NW;
    for i in 0..size_4d {
        let base = i*NUM_FIELDS;
        let nu = fields.data[base+2];
        fields.data[base+5] += TORSION_STRENGTH * nu * DT * 1e-6;
        fields.data[base+6] += TORSION_STRENGTH * nu * DT * 1e-6;
        fields.data[base+4] += (fields.data[base+5] + fields.data[base+6]) * DT * 1e-2;
    }
}

fn update_chiral_odd(fields: &mut FieldData) {
    let size_4d = NX * NY * NZ * NW;
    for idx_3d in 0..size_4d {
        let base = idx_3d * NUM_FIELDS;

        // Fetch relevant fields
        let torsion_xx = fields.data[base + 5];
        let torsion_xy = fields.data[base + 6];
        let metric = fields.data[base + 4];
        let neutrino_density = fields.data[base + 2];

        // Dynamic update for chiral odd component
        let delta_chiral = TORSION_STRENGTH * (torsion_xx - torsion_xy)
            - 0.01 * metric
            + 1e-3 * neutrino_density;

        //fields.data[base + 8] += delta_chiral * DT;
        let casimir_effect = CASIMIR_COUPLING / ((r + DX).powi(4));
        fields.data[base + 8] += casimir_effect * DT;
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

fn compute_chiral_avg(fields: &FieldData) -> f64 {
    let size_4d = NX * NY * NZ * NW;
    let mut sum_chiral = 0.0;
    for idx_3d in 0..size_4d {
        let base = idx_3d * NUM_FIELDS;
        sum_chiral += fields.data[base + 8];
    }
    sum_chiral / size_4d as f64
}

fn main() -> Result<(),Box<dyn Error>> {

    let current_dir = env::current_dir()?;
    println!("Current working directory: {:?}", current_dir);

    // Attempt to load meson data
    let mesons = match load_mesons("mesons.csv") {
        Ok(m) => {
            if m.is_empty() {
                println!("Warning: mesons.csv loaded but no mesons found!");
            } else {
                println!("Meson data loaded successfully. Found {} mesons:", m.len());
                for mes in &m {
                    println!(" - {}: mass={} MeV/c², lifetime={} s", mes.name, mes.mass, mes.lifetime);
                }
            }
            m
        },
        Err(e) => {
            eprintln!("No meson data found or failed to load mesons.csv. Error was: {}", e);
            Vec::new()
        }
    };    

    let mut fields = FieldData::new();
    update_chiral_odd(&mut fields);
    let mut scale_factor = ((8.0 * PI.powi(2)) * (r.powi(2))) / 15.0 * (-((8.0 * PI.powi(2)) * (r.powi(2))) / 15.0).sqrt();

    let mut file = BufWriter::new(File::create("results.csv")?);
    writeln!(file,"time,a,avg_photon,avg_axion,avg_neutrino,avg_energy")?;

    for step in 0..STEPS {
        measure_preserving_transform(&mut fields);
        update_ec_torsion_metric(&mut fields);
        time_step_rk4(&mut fields);

        // Meson decay if available
        if !mesons.is_empty() {
            let mut rng = rand::thread_rng();
            let m_idx = rng.gen_range(0..mesons.len());
            if let Some(prods) = simulate_decay(&mesons[m_idx]) {
                // Add neutrinos or photons uniformly
                let size_4d = NX*NY*NZ*NW;
                for i in 0..size_4d {
                    let base = i*NUM_FIELDS;
                    for p in &prods {
                        // Print confirmation that decay products are applied
                        println!("Injected decay product {} at step {}", p, step);
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

        let (avg_ph, avg_ax, avg_nu, avg_e) = compute_averages(&fields);
        let avg_chiral = compute_chiral_avg(&fields);

        scale_factor += DT * scale_factor * avg_e.sqrt() * avg_chiral.sqrt();

        let time = step as f64 * DT;

        writeln!(file, "{},{},{},{},{},{},{}", time, scale_factor, avg_ph, avg_ax, avg_nu, avg_e, avg_chiral)?;

        if step % OUTPUT_INTERVAL == 0 {
            println!("Step {}: a={:.5}, ph={:.5}, ax={:.5}, nu={:.5}, E={:.5}",
                     step, scale_factor, avg_ph, avg_ax, avg_nu, avg_e);
            write_snapshot(step,&fields)?;
        }
    }

    println!("Simulation complete. Results in results.csv and snapshots/");
    Ok(())
}
