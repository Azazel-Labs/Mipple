//! Learned rankers for the localized RD search, one per format: which far earlier blocks of the
//! chunk get the full candidate search. A ranker only proposes blocks; the RD search still picks
//! the encoding from their candidates, so a bad ranking costs compression, never correctness.
//!
//! Each is a small MLP trained by `scripts/train_ranker*.py` on CC0 Poly Haven textures
//! (`scripts/fetch_polyhaven.py`), never on the Kodak or Oodle benchmark images, from the rows
//! `MIPPLE_LOCAL_DUMP` writes.

pub mod bc4;
pub mod bc5;
pub mod bc7;
