mod! SIMPLE-NAT {
  [ Nat ]
  op 0 : -> Nat
  op s : Nat -> Nat
}

mod! NAT+ {
  pr(SIMPLE-NAT)
  op _+_ : Nat Nat -> Nat
  eq 0 + M:Nat = M .
  eq s (N:Nat) + M:Nat = s (N + M) .
}
