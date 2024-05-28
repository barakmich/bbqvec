use bbqvec;

#[test]
fn creates_a_vector() {
    let v = bbqvec::create_random_vector(20);
    assert_eq!(v.len(), 20);
}

#[test]
fn full_table_scan() {
    let vecs = create_vector_set(20, 200);
}
