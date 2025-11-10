# Bugs that took a bit to fix


I originally scaled the weights as
wts = permutedims(q, (2,1,3))  ⊠ k .* sqrt(head_size)

When running the forward model only I got an error
```
❯ julia --project=. --threads=auto src/training_enzyme_clean_v1.jl
Precompiling NanoLux...
  1 dependency successfully precompiled in 3 seconds. 274 already precompiled.
┌ Warning: Mixed-Precision `matmul_cpu_fallback!` detected and Octavian.jl cannot be used for this set of inputs (C [Matrix{Float64}]: A [Matrix{Float32}] x B [Matrix{Float64}]). Falling back to generic implementation. This may be slow.
└ @ LuxLib.Impl ~/.julia/packages/LuxLib/R8Czx/src/impl/matmul.jl:190
loc(callsite("transpose"("/Users/ralph/.julia/packages/Reactant/rauLT/src/Ops.jl":589:0) at "traced_call/call"("/Users/ralph/.julia/packages/Reactant/rauLT/src/ControlFlow.jl":8:0))): error: 'stablehlo.transpose' op requires compatible element types for all operands and results
┌ Error: Compilation failed, MLIR module written to /var/folders/01/5dcdxf8139bgcqz5z4qfk8lc0000gn/T/reactant_HR3aCR/module_000_pUIX_post_all_pm.mlir
└ @ Reactant.MLIR.IR ~/.julia/packages/Reactant/rauLT/src/mlir/IR/Pass.jl:119
ERROR: LoadError: "failed to run pass manager on module"
Stacktrace:
  [1] run!(pm::Reactant.MLIR.IR.PassManager, mod::Reactant.MLIR.IR.Module, key::String)
    @ Reactant.MLIR.IR ~/.julia/packages/Reactant/rauLT/src/mlir/IR/Pass.jl:163
  [2] run_pass_pipeline!(mod::Reactant.MLIR.IR.Module, pass_pipeline::String, key::String; enable_verifier::Bool)
    @ Reactant.Compiler ~/.julia/packages/Reactant/rauLT/src/Compiler.jl:1320
  [3] run_pass_pipeline!
    @ ~/.julia/packages/Reactant/rauLT/src/Compiler.jl:1315 [inlined]
```

It took a bit to find that the square-root had to be directly casted to Float32:

wts = permutedims(q, (2,1,3))  ⊠ k .* Float32(sqrt(head_size)) 



