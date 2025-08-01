#!/bin/bash
cd /root/App_Datos_VillarrealCF
git pull origin main
git add .
if ! git diff --cached --quiet; then
    git commit -m "Actualización automática $(date)"
    git push origin main
    echo "$(date): Cambios subidos" >> /var/log/villarreal-updates.log
else
    echo "$(date): Sin cambios" >> /var/log/villarreal-updates.log
fi
sudo systemctl restart villarreal-app.service
