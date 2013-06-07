#!/usr/bin/env ruby
require 'igor'

$datasets="/sampa/home/bdmyers/graph_datasets"

Igor do
  
  database 'join.omp.db', :triangles

  command 'srun -p grappa ./triangles_parallel %{fin} %{ppn}'

  sbatch_flags << "--time=60"
  
  params {
    nnode       1
    ppn         2
    fin         ""
    tag         'none'
  }

 run {
    fin "#{$datasets}/berkstan/web-BerkStan.txt"
 }
   
  
  expect :triangles_runtime
 
  #$filtered = results{|t| t.select(:id, :nnode, :ppn, :tree, :run_at, :search_runtime) }
    
  interact # enter interactive mode
end
