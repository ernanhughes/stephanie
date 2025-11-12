# From repo root
Compress-Archive -Path stephanie\components\nexus, stephanie\services, stephanie\utils, stephanie\constants.py `
  -DestinationPath nexus_v2_code.zip -Force

Compress-Archive -Path runs\nexus_vpm\8242-baseline\manifest.json, `
                        runs\nexus_vpm\8242-baseline\graph.json, `
                        runs\nexus_vpm\8242-baseline\run_metrics.json, `
                        runs\nexus_vpm\8242-targeted\manifest.json, `
                        runs\nexus_vpm\8242-targeted\graph.json, `
                        runs\nexus_vpm\8242-targeted\run_metrics.json `
  -DestinationPath nexus_v2_sample_runs.zip -Force
