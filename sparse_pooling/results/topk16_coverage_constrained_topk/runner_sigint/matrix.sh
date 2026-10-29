PY=/home/evga/.conda/envs/env_isaaclab/bin/python; T=/home/evga/Projects/pamachine/GranuleLayer-tkc/sparse_pooling/tests
for disp in dfl ign; do for mode in startup ready; do for load in 0 6; do
  echo "== $disp $mode load$load $(date +%T)"
  $PY stress.py $T runs/m_${disp}_${mode}_${load} 30 $mode $load $disp > m_${disp}_${mode}_${load}.jsonl 2>&1
  tail -1 m_${disp}_${mode}_${load}.jsonl
done; done; done
echo MATRIX DONE
