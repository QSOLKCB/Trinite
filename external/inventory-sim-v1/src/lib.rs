//! Bounded integer inventory environment, separate from native Trinite learning.
pub mod json;
use json::Value;
use std::collections::BTreeMap;

pub const SCHEMA: &str = "trinite.inventory-request.v1";
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Request {
    pub capacity: u16,
    pub initial_stock: u16,
    pub demand: Vec<u16>,
    pub orders: Vec<u16>,
    pub lead_times: Vec<u16>,
    pub supplier_available: Vec<bool>,
    pub observe_ticks: u16,
    pub observed: [bool; 8],
}
fn object(value: Value) -> Result<BTreeMap<String, Value>, String> {
    match value {
        Value::Object(v) => Ok(v),
        _ => Err("expected object".into()),
    }
}
fn take(fields: &mut BTreeMap<String, Value>, key: &str) -> Result<Value, String> {
    fields.remove(key).ok_or_else(|| format!("missing {key}"))
}
fn integer(value: Value, min: i64, max: i64) -> Result<u16, String> {
    match value {
        Value::Int(n) if (min..=max).contains(&n) => Ok(n as u16),
        _ => Err("integer outside admitted bounds".into()),
    }
}
fn array(value: Value) -> Result<Vec<Value>, String> {
    match value {
        Value::Array(v) => Ok(v),
        _ => Err("expected array".into()),
    }
}
fn numbers(value: Value) -> Result<Vec<u16>, String> {
    array(value)?
        .into_iter()
        .map(|v| integer(v, 0, 32))
        .collect()
}
fn booleans(value: Value) -> Result<Vec<bool>, String> {
    array(value)?
        .into_iter()
        .map(|v| match v {
            Value::Bool(b) => Ok(b),
            _ => Err("expected boolean".into()),
        })
        .collect()
}
impl Request {
    pub fn parse(bytes: &[u8]) -> Result<Self, String> {
        let mut f = object(json::parse(bytes)?)?;
        if take(&mut f, "schema")? != Value::Str(SCHEMA.into()) {
            return Err("unknown request schema".into());
        }
        let request = Self {
            capacity: integer(take(&mut f, "capacity")?, 1, 128)?,
            initial_stock: integer(take(&mut f, "initial_stock")?, 0, 128)?,
            demand: numbers(take(&mut f, "demand")?)?,
            orders: numbers(take(&mut f, "orders")?)?,
            lead_times: numbers(take(&mut f, "lead_times")?)?,
            supplier_available: booleans(take(&mut f, "supplier_available")?)?,
            observe_ticks: integer(take(&mut f, "observe_ticks")?, 1, 255)?,
            observed: booleans(take(&mut f, "observed")?)?
                .try_into()
                .map_err(|_| "observed must have eight booleans")?,
        };
        if !f.is_empty() {
            return Err("unknown field".into());
        }
        request.validate()?;
        Ok(request)
    }
    pub fn validate(&self) -> Result<(), String> {
        let n = self.demand.len();
        if !(1..=128).contains(&self.capacity)
            || self.initial_stock > self.capacity
            || !(2..=256).contains(&n)
            || self.orders.len() != n
            || self.lead_times.len() != n
            || self.supplier_available.len() != n
            || self.observe_ticks == 0
            || usize::from(self.observe_ticks) >= n
            || self
                .demand
                .iter()
                .chain(&self.orders)
                .chain(&self.lead_times)
                .any(|&x| x > 32)
        {
            return Err("invalid inventory bounds or schedule lengths".into());
        }
        Ok(())
    }
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Batch {
    pub ordered_tick: u16,
    pub due_tick: u16,
    pub quantity: u16,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tick {
    pub tick: u16,
    pub ordered: u16,
    pub delivered: u16,
    pub received: u16,
    pub discarded: u16,
    pub demand: u16,
    pub sales: u16,
    pub lost: u16,
    pub stock: u16,
    pub pending_units: u16,
    pub supplier_available: bool,
}
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Summary {
    pub ordered: u32,
    pub delivered: u32,
    pub received: u32,
    pub discarded: u32,
    pub demand: u32,
    pub sales: u32,
    pub lost: u32,
    pub stockout_ticks: u16,
}
impl Summary {
    fn add(&mut self, row: &Tick) {
        self.ordered += u32::from(row.ordered);
        self.delivered += u32::from(row.delivered);
        self.received += u32::from(row.received);
        self.discarded += u32::from(row.discarded);
        self.demand += u32::from(row.demand);
        self.sales += u32::from(row.sales);
        self.lost += u32::from(row.lost);
        self.stockout_ticks += u16::from(row.lost > 0);
    }
    fn json(&self) -> String {
        format!("{{\"ordered\":{},\"delivered\":{},\"received\":{},\"discarded\":{},\"demand\":{},\"sales\":{},\"lost\":{},\"stockout_ticks\":{}}}",
            self.ordered,self.delivered,self.received,self.discarded,self.demand,self.sales,self.lost,self.stockout_ticks)
    }
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Simulation {
    pub rows: Vec<Tick>,
    pub summary: Summary,
    pub forecast: Summary,
    pub final_stock: u16,
    pub pending: Vec<Batch>,
}
// One transition owns both observations and truth. Observe executes prefix only.
fn run(request: &Request, steps: usize) -> Simulation {
    let mut stock = request.initial_stock;
    let mut pending: Vec<Batch> = Vec::new();
    let mut rows = Vec::new();
    let mut summary = Summary::default();
    let mut forecast = Summary::default();
    for t in 0..steps {
        let tick = t as u16;
        let ordered = request.orders[t];
        if ordered > 0 {
            pending.push(Batch {
                ordered_tick: tick,
                due_tick: tick + request.lead_times[t],
                quantity: ordered,
            });
        }
        let mut delivered = 0u16;
        if request.supplier_available[t] {
            pending.retain(|batch| {
                if batch.due_tick <= tick {
                    delivered += batch.quantity;
                    false
                } else {
                    true
                }
            });
        }
        let received = delivered.min(request.capacity - stock);
        stock += received;
        let demand = request.demand[t];
        let sales = demand.min(stock);
        stock -= sales;
        let row = Tick {
            tick,
            ordered,
            delivered,
            received,
            discarded: delivered - received,
            demand,
            sales,
            lost: demand - sales,
            stock,
            pending_units: pending.iter().map(|b| b.quantity).sum(),
            supplier_available: request.supplier_available[t],
        };
        summary.add(&row);
        if tick >= request.observe_ticks {
            forecast.add(&row);
        }
        rows.push(row);
    }
    Simulation {
        rows,
        summary,
        forecast,
        final_stock: stock,
        pending,
    }
}
pub fn simulate(request: &Request) -> Result<Simulation, String> {
    request.validate()?;
    Ok(run(request, request.demand.len()))
}
pub fn observation(request: &Request) -> Result<String, String> {
    request.validate()?;
    let simulation = run(request, usize::from(request.observe_ticks));
    let mask = request
        .observed
        .iter()
        .map(bool::to_string)
        .collect::<Vec<_>>()
        .join(",");
    let rows = simulation
        .rows
        .iter()
        .map(|r| {
            let raw = [
                r.stock,
                r.demand,
                r.sales,
                r.lost,
                r.received,
                r.discarded,
                r.ordered,
                r.pending_units,
            ];
            let values = raw
                .iter()
                .zip(request.observed)
                .map(|(v, known)| if known { v.to_string() } else { "0".into() })
                .collect::<Vec<_>>()
                .join(",");
            format!(
                "{{\"tick\":{},\"supplier_available\":{},\"values\":[{}],\"observed\":[{}]}}",
                r.tick, r.supplier_available, values, mask
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    Ok(format!("{{\"schema\":\"trinite.inventory-observation.v1\",\"capacity\":{},\"observe_ticks\":{},\"horizon\":{},\"rows\":[{}]}}\n",
        request.capacity,request.observe_ticks,request.demand.len()-usize::from(request.observe_ticks),rows))
}
pub fn privileged_json(request: &Request) -> Result<String, String> {
    let s = simulate(request)?;
    let rows = s.rows.iter().map(|r| format!("{{\"tick\":{},\"ordered\":{},\"delivered\":{},\"received\":{},\"discarded\":{},\"demand\":{},\"sales\":{},\"lost\":{},\"stock\":{},\"pending_units\":{},\"supplier_available\":{}}}",
        r.tick,r.ordered,r.delivered,r.received,r.discarded,r.demand,r.sales,r.lost,r.stock,r.pending_units,r.supplier_available)).collect::<Vec<_>>().join(",");
    let pending = s
        .pending
        .iter()
        .map(|b| {
            format!(
                "{{\"ordered_tick\":{},\"due_tick\":{},\"quantity\":{}}}",
                b.ordered_tick, b.due_tick, b.quantity
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    Ok(format!("{{\"schema\":\"trinite.inventory-ground-truth.v1\",\"privileged\":true,\"capacity\":{},\"observe_ticks\":{},\"steps\":{},\"initial_stock\":{},\"rows\":[{}],\"summary\":{},\"forecast\":{},\"final_stock\":{},\"pending\":[{}]}}\n",
        request.capacity,request.observe_ticks,request.demand.len(),request.initial_stock,rows,s.summary.json(),s.forecast.json(),s.final_stock,pending))
}
#[cfg(test)]
mod tests {
    use super::*;
    fn example() -> Request {
        Request::parse(include_bytes!("../example-request.json")).unwrap()
    }
    #[test]
    fn receipt_before_demand_and_zero_lead() {
        let mut r = example();
        r.capacity = 3;
        r.initial_stock = 3;
        r.demand = vec![3, 1];
        r.orders = vec![3, 1];
        r.lead_times = vec![0, 0];
        r.supplier_available = vec![true; 2];
        r.observe_ticks = 1;
        let s = simulate(&r).unwrap();
        assert_eq!(
            (s.rows[0].delivered, s.rows[0].discarded, s.rows[0].sales),
            (3, 3, 3)
        );
        assert_eq!(
            (s.rows[1].received, s.rows[1].sales, s.final_stock),
            (1, 1, 0)
        );
        assert_eq!(s.summary.stockout_ticks, 0);
    }
    #[test]
    fn unavailable_supplier_holds_due_orders_then_releases_all() {
        let s = simulate(&example()).unwrap();
        assert_eq!(s.rows[1].delivered, 0);
        assert_eq!(s.rows[2].pending_units, 7);
        assert_eq!((s.rows[3].delivered, s.rows[3].received), (7, 7));
        assert_eq!(
            s.pending,
            vec![Batch {
                ordered_tick: 5,
                due_tick: 8,
                quantity: 2
            }]
        );
    }
    #[test]
    fn lost_sales_are_not_backorders_or_empty_stock_flags() {
        let mut r = example();
        r.initial_stock = 0;
        r.orders.fill(0);
        r.demand = vec![0, 1, 0, 3, 0, 0];
        let s = simulate(&r).unwrap();
        assert_eq!(s.summary.lost, 4);
        assert_eq!(s.summary.stockout_ticks, 2);
        assert_eq!(s.summary.sales, 0);
    }
    #[test]
    fn masks_and_suffix_do_not_change_prefix_dynamics() {
        let mut r = example();
        let s = simulate(&r).unwrap();
        r.observed = [false; 8];
        assert_eq!(simulate(&r).unwrap(), s);
        let text = observation(&r).unwrap();
        assert!(text.contains("\"values\":[0,0,0,0,0,0,0,0]"));
        for t in usize::from(r.observe_ticks)..r.demand.len() {
            r.demand[t] = 32;
            r.orders[t] = 32;
            r.lead_times[t] = 32;
            r.supplier_available[t] = false;
        }
        assert_eq!(observation(&r).unwrap(), text);
    }
    #[test]
    fn bounds_and_conservation_with_all_pending_then_mass_receipt() {
        let mut r = Request {
            capacity: 128,
            initial_stock: 128,
            demand: vec![32; 256],
            orders: vec![32; 256],
            lead_times: vec![0; 256],
            supplier_available: vec![false; 256],
            observe_ticks: 255,
            observed: [true; 8],
        };
        let s = simulate(&r).unwrap();
        assert_eq!(s.pending.len(), 256);
        assert_eq!(s.rows[255].pending_units, 8192);
        r.supplier_available[255] = true;
        let s = simulate(&r).unwrap();
        assert_eq!(s.rows[255].delivered, 8192);
        assert_eq!(s.rows[255].discarded, 8064);
        assert_eq!(
            128 + s.summary.ordered,
            s.summary.sales + s.summary.discarded + u32::from(s.final_stock)
        );
        assert_eq!(s.summary.demand, s.summary.sales + s.summary.lost);
    }
    #[test]
    fn due_boundary_is_not_silently_flushed() {
        let mut r = example();
        r.lead_times.fill(32);
        let s = simulate(&r).unwrap();
        assert_eq!(s.summary.delivered, 0);
        assert_eq!(s.pending.len(), 4);
        assert_eq!(s.pending.last().unwrap().due_tick, 37);
    }
    #[test]
    fn direct_api_rejects_bounds() {
        let mut r = example();
        r.capacity = 0;
        assert!(simulate(&r).is_err());
        assert!(observation(&r).is_err());
        r = example();
        r.lead_times[0] = 33;
        assert!(simulate(&r).is_err());
        r = example();
        r.orders.pop();
        assert!(simulate(&r).is_err());
        r = example();
        r.initial_stock = 9;
        assert!(simulate(&r).is_err());
        r = example();
        r.observe_ticks = 6;
        assert!(observation(&r).is_err());
    }
    #[test]
    fn unknown_missing_and_wrong_type_rejected() {
        let source = String::from_utf8(include_bytes!("../example-request.json").to_vec()).unwrap();
        for text in [
            source.replace("\"capacity\":8", "\"capacity\":true"),
            source.replace("\"capacity\":8", "\"capacity\":8,\"other\":0"),
            source.replace("\"capacity\":8,", ""),
            source.replace("\"observe_ticks\":3", "\"observe_ticks\":0"),
        ] {
            assert!(Request::parse(text.as_bytes()).is_err());
        }
    }
}
