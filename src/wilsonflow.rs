use crate::observables::Observable;

#[allow(non_camel_case_types, dead_code)]
pub enum WfObservables {
    t,
    E,
    t2_E,
    Esym,
    t2_Esym,
    TC,
}
impl WfObservables {
    //(t,E,t2*E,Esym,t2*Esym,TC)
    pub fn get_offset(&self) -> usize {
        match self {
            WfObservables::t => 0,
            WfObservables::E => 1,
            WfObservables::t2_E => 2,
            WfObservables::Esym => 3,
            WfObservables::t2_Esym => 4,
            WfObservables::TC => 5,
        }
    }
}

pub struct WilsonFlow {
    t: Vec<f64>,
    t2_Esym: Observable,
    TC: Observable,
}
