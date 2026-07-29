use std::sync::OnceLock;

pub struct Zobrist {
    pub pieces: [[u64; 256]; 196],
    pub turn: [u64; 4],
    pub ep_squares: [u64; 196],
    pub castling: [[u64; 4]; 4],
}

static ZOBRIST: OnceLock<Zobrist> = OnceLock::new();

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
}

pub fn get_zobrist() -> &'static Zobrist {
    ZOBRIST.get_or_init(|| {
        let mut rng = SplitMix64::new(0x123456789ABCDEF0);
        let mut pieces = [[0u64; 256]; 196];
        for sq in 0..196 {
            for p in 0..256 {
                pieces[sq][p] = rng.next();
            }
        }
        let mut ep_squares = [0u64; 196];
        for i in 0..196 {
            ep_squares[i] = rng.next();
        }
        let mut castling = [[0u64; 4]; 4];
        for c in 0..4 {
            for r in 0..4 {
                castling[c][r] = rng.next();
            }
        }
        Zobrist {
            pieces,
            turn: [rng.next(), rng.next(), rng.next(), rng.next()],
            ep_squares,
            castling,
        }
    })
}