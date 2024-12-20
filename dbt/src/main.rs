use std::f64::consts::{E, PI};

//------------------------------------------------------------
// Parameters
//------------------------------------------------------------
const NX: usize = 4;
const NY: usize = 4;
const NZ: usize = 4; 
const NW: usize = 2; // Additional dimensions (5D total)
const NV: usize = 2; // Additional dimensions
const D: usize = 5; // 5D spacetime

const DT: f64 = (1.0)/137.0 * PI * E;
const STEPS: usize = 10;

// Physical parameters (tweak these as needed)
const LAMBDA: f64 = 1.0;
const V0: f64 = 1.0;
const GAUGE_DAMP: f64 = 0.1;
const FERMION_DAMP: f64 = 0.1;
const TORSION_COUPLING: f64 = 0.001;
const TORSION_DAMP: f64 = 0.05;
const SPINOR_CORR_FACTOR: f64 = 0.01;

const NUM_SPINOR: usize = 4;
const NUM_FIELDS: usize = 6; // Fields: Φ, A0, Psi0, Psi1, Psi2, Psi3

//------------------------------------------------------------
// Data structure
//------------------------------------------------------------
struct FieldConfig {
    data: Vec<f64>,       // Holds scalar, gauge, and fermion fields
    torsion: Vec<f64>,    // T_{abc}
    contorsion: Vec<f64>, // K_{abc}
}

impl FieldConfig {
    fn total_points() -> usize {
        NX*NY*NZ*NW*NV
    }

    fn new() -> Self {
        let size = Self::total_points()*NUM_FIELDS;
        let mut data = vec![0.0; size];
        for val in data.iter_mut() {
            *val = (rand01()-0.5)*0.01;
        }

        let t_size = Self::total_points()*D*D*D;
        let mut torsion = vec![0.0; t_size];
        let mut contorsion = vec![0.0; t_size];

        // Initialize torsion to small random values
        for val in torsion.iter_mut() {
            *val = (rand01()-0.5)*0.001;
        }

        Self { data, torsion, contorsion }
    }

    fn fidx(x:usize,y:usize,z:usize,w:usize,v:usize,f:usize) -> usize {
        f + NUM_FIELDS*(x + NX*(y + NY*(z + NZ*(w + NW*v))))
    }

    fn get_field(&self,x:usize,y:usize,z:usize,w:usize,v:usize,f:usize) -> f64 {
        self.data[Self::fidx(x,y,z,w,v,f)]
    }

    fn set_field(&mut self,x:usize,y:usize,z:usize,w:usize,v:usize,f:usize,val:f64) {
        let idx = Self::fidx(x,y,z,w,v,f);
        self.data[idx] = val;
    }

    fn tidx(x:usize,y:usize,z:usize,w:usize,v:usize,a:usize,b:usize,c:usize) -> usize {
        a + D*(b + D*(c + D*(x + NX*(y + NY*(z + NZ*(w + NW*v))))))
    }

    fn get_torsion(&self,x:usize,y:usize,z:usize,w:usize,v:usize,a:usize,b:usize,c:usize) -> f64 {
        self.torsion[Self::tidx(x,y,z,w,v,a,b,c)]
    }

    fn set_torsion(&mut self,x:usize,y:usize,z:usize,w:usize,v:usize,a:usize,b:usize,c:usize,val:f64) {
        let idx = Self::tidx(x,y,z,w,v,a,b,c);
        self.torsion[idx] = val;
    }

    fn set_contorsion(&mut self,x:usize,y:usize,z:usize,w:usize,v:usize,a:usize,b:usize,c:usize,val:f64) {
        let idx = Self::tidx(x,y,z,w,v,a,b,c);
        self.contorsion[idx] = val;
    }

    fn get_contorsion(&self,x:usize,y:usize,z:usize,w:usize,v:usize,a:usize,b:usize,c:usize) -> f64 {
        self.contorsion[Self::tidx(x,y,z,w,v,a,b,c)]
    }
}

//------------------------------------------------------------
// Utilities
//------------------------------------------------------------
fn rand01() -> f64 {
    static mut SEED: u64 = 12345;
    unsafe {
        SEED = SEED.wrapping_mul(6364136223846793005).wrapping_add(1);
        let r = (SEED >> 33) as u32;
        (r as f64)/(std::u32::MAX as f64)
    }
}

//------------------------------------------------------------
// Physics routines
//------------------------------------------------------------
fn potential_derivative(phi:f64) -> f64 {
    let diff = phi*phi - V0*V0;
    4.0*LAMBDA*diff*phi
}

// Compute contorsion from torsion:
// K_{abc} = 1/2 (T_{abc} + T_{cab} - T_{bca})
fn compute_contorsion(fc: &mut FieldConfig) {
    for v in 0..NV {
        for w in 0..NW {
            for z in 0..NZ {
                for y in 0..NY {
                    for x in 0..NX {
                        for a in 0..D {
                            for b in 0..D {
                                for c in 0..D {
                                    let tabc = fc.get_torsion(x,y,z,w,v,a,b,c);
                                    let tcab = fc.get_torsion(x,y,z,w,v,c,a,b);
                                    let tbca = fc.get_torsion(x,y,z,w,v,b,c,a);
                                    let k_abc = 0.5*(tabc + tcab - tbca);
                                    fc.set_contorsion(x,y,z,w,v,a,b,c,k_abc);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// Placeholder for full Dirac equation with vielbein e^a_μ and gamma matrices γ^a:
// Currently, we just approximate with contorsion trace.
// In a full simulation:
//   - Compute e^a_μ from metric
//   - Compute ω_μ^{ab} from contorsion and metric
//   - Use γ^a and e^a_μ to form γ^μ = e^μ_a γ^a
//   - Solve (i γ^μ D_μ - m)ψ = 0 properly.
// For now, we keep the simple approximation:
fn fermion_update(psi:&[f64;4], fc:&FieldConfig, x:usize,y:usize,z:usize,w:usize,v:usize) -> [f64;4] {
    let mut trace = 0.0;
    for μ in 0..D {
        for α in 0..D {
            trace += fc.get_contorsion(x,y,z,w,v,μ,α,α);
        }
    }

    let mut out = [0.0;4];
    for i in 0..4 {
        let torsion_corr = SPINOR_CORR_FACTOR*trace*psi[i];
        out[i] = -FERMION_DAMP*psi[i] + torsion_corr;
    }
    out
}

// Torsion update from fermion density
fn torsion_update(psi:&[f64;4], fc:&mut FieldConfig, x:usize,y:usize,z:usize,w:usize,v:usize) {
    let fermion_density = psi.iter().map(|x| x*x).sum::<f64>();
    for a in 0..D {
        for b in 0..D {
            for c in 0..D {
                let t_old = fc.get_torsion(x,y,z,w,v,a,b,c);
                let dt = TORSION_COUPLING*fermion_density - TORSION_DAMP*t_old;
                let t_new = t_old + DT*dt;
                fc.set_torsion(x,y,z,w,v,a,b,c,t_new);
            }
        }
    }
}

fn scalar_update(phi:f64) -> f64 {
    let dVdphi = potential_derivative(phi);
    -dVdphi
}

fn gauge_update(a0:f64) -> f64 {
    -GAUGE_DAMP*a0
}

fn relaxation_step(fc: &mut FieldConfig) {
    compute_contorsion(fc);

    let mut new_data = fc.data.clone();

    for v in 0..NV {
        for w in 0..NW {
            for z in 0..NZ {
                for y in 0..NY {
                    for x in 0..NX {
                        let phi = fc.get_field(x,y,z,w,v,0);
                        let a0 = fc.get_field(x,y,z,w,v,1);
                        let psi = [
                            fc.get_field(x,y,z,w,v,2),
                            fc.get_field(x,y,z,w,v,3),
                            fc.get_field(x,y,z,w,v,4),
                            fc.get_field(x,y,z,w,v,5),
                        ];

                        let dphi_dt = scalar_update(phi);
                        let da0_dt = gauge_update(a0);
                        let dpsi_dt = fermion_update(&psi, fc, x,y,z,w,v);

                        let base = FieldConfig::fidx(x,y,z,w,v,0);
                        new_data[base+0] = phi + DT*dphi_dt;
                        new_data[base+1] = a0 + DT*da0_dt;
                        for i in 0..4 {
                            new_data[base+2+i] = psi[i] + DT*dpsi_dt[i];
                        }

                        torsion_update(&psi, fc, x,y,z,w,v);
                    }
                }
            }
        }
    }

    fc.data = new_data;
}

// Save snapshots to visualize
fn save_snapshot(fc: &FieldConfig, step: usize) {
    let fields_file = format!("fields_{:05}.bin", step);
    let torsion_file = format!("torsion_{:05}.bin", step);

    // Save fields
    {
        use std::io::Write;
        let mut f = std::fs::File::create(fields_file).expect("Can't create fields file");
        // Just write raw
        for val in fc.data.iter() {
            f.write_all(&val.to_le_bytes()).unwrap();
        }
    }

    // Save torsion
    {
        use std::io::Write;
        let mut f = std::fs::File::create(torsion_file).expect("Can't create torsion file");
        for val in fc.torsion.iter() {
            f.write_all(&val.to_le_bytes()).unwrap();
        }
    }
}

fn main() {
    let mut fc = FieldConfig::new();

    for step in 0..STEPS {
        relaxation_step(&mut fc);

        if step % (STEPS/10) == 0 {
            // Compute average phi
            let size = NX*NY*NZ*NW*NV;
            let mut sum_phi = 0.0;
            for v in 0..NV {
                for w in 0..NW {
                    for z in 0..NZ {
                        for y in 0..NY {
                            for x in 0..NX {
                                sum_phi += fc.get_field(x,y,z,w,v,0);
                            }
                        }
                    }
                }
            }
            let avg_phi = sum_phi/(size as f64);
            println!("Step {}: avg_phi = {}", step, avg_phi);

            // Save snapshot occasionally
            save_snapshot(&fc, step);
        }
    }

    println!("Simulation complete. Data saved in fields_XXXXX.bin and torsion_XXXXX.bin.");
}
