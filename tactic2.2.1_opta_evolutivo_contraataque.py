import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
import os
import warnings
from PIL import Image as PILImage
from collections import Counter
from difflib import SequenceMatcher
from datetime import timedelta

warnings.filterwarnings('ignore')

class AnalizadorContraataquesOpta:
    def __init__(self, events_path="extraccion_opta/datos_opta_parquet/match_events.parquet"):
        self.events_path = events_path
        self.df = None
        self.team_stats = None
        self.all_matches_df = None
        
    def similarity(self, a, b):
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    def resize_image_to_fixed_size(self, image, target_size=100):
        """Redimensiona imagen a un tamaño fijo manteniendo proporción y centrado (estilo ABP3)"""
        try:
            from PIL import Image as PILImage
            if image.dtype != np.uint8: image = (image * 255).astype(np.uint8)
            pil_image = PILImage.fromarray(image, 'RGBA' if image.shape[2] == 4 else 'RGB')
            pil_image.thumbnail((target_size, target_size), PILImage.Resampling.LANCZOS)
            # Lienzo cuadrado transparente
            square_image = PILImage.new('RGBA', (target_size, target_size), (0, 0, 0, 0))
            x_offset = (target_size - pil_image.width) // 2
            y_offset = (target_size - pil_image.height) // 2
            square_image.paste(pil_image, (x_offset, y_offset))
            return np.array(square_image) / 255.0
        except: return image

    def convert_to_grayscale(self, image):
        try:
            if image.shape[2] == 4:
                gray = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
                gray_img = np.zeros_like(image)
                for i in range(3): gray_img[..., i] = gray
                gray_img[..., 3] = image[..., 3] * 0.5
                return gray_img
            return image
        except: return image
    
    def debug_analisis(self, match_id_test=None):
        if self.df is None: return
        m_id = match_id_test if match_id_test else self.df['Match ID'].iloc[0]
        test_df = self.df[self.df['Match ID'] == m_id]
        rec_ids = [7, 8, 44, 49, 52, 54, 59, 74]
        recuperaciones = test_df[test_df['typeId'].astype(int).isin(rec_ids)]

    def find_team_logo_by_similarity(self, equipo, grayscale=False):
        """Busca el escudo por similitud y lo redimensiona (estilo ABP3)"""
        if not os.path.exists('assets/escudos'): return None
        archivos = [f for f in os.listdir('assets/escudos') if f.endswith('.png')]
        eq_clean = equipo.lower().replace(' ', '').replace('cf', '').replace('fc', '').replace('real', '').replace('rcd', '')
        best_match, max_sim = None, 0
        for f in archivos:
            f_clean = f.replace('.png', '').lower().replace('_', '').replace('cf', '').replace('fc', '').replace('real', '').replace('rcd', '')
            sim = self.similarity(eq_clean, f_clean)
            if sim > max_sim:
                max_sim, best_match = sim, f
        if best_match and max_sim > 0.35:
            img = plt.imread(f"assets/escudos/{best_match}")
            img = self.resize_image_to_fixed_size(img, target_size=100)
            if grayscale: 
                img = self.convert_to_grayscale(img)
            return img
        return None

    def load_data(self):
        pass
        self.df = pd.read_parquet(self.events_path)
        self.df['timeStamp'] = pd.to_datetime(self.df['timeStamp'], format='ISO8601')
        self.df = self.df.sort_values(['Match ID', 'timeStamp']).reset_index()
        self._analizar_liga()

    def _analizar_liga(self):
        pass
        self.df = self.df.sort_values(['Match ID', 'periodId', 'timeStamp']).reset_index(drop=True)
        self.df['index'] = self.df.index
        rec_ids = [7, 8, 44, 49, 52, 54, 59, 74]
        all_results = []

        for (m_id, t_name), group in self.df.groupby(['Match ID', 'Team Name']):
            c_count, g_count = 0, 0
            origins = []
            for _, row in group.iterrows():
                if int(row['typeId']) in rec_ids:
                    sx = row['x']
                    if sx > 70: continue
                    max_d = 12 if sx < 25 else (10 if sx < 50 else 5)
                    start_idx = row['index']
                    seq = self.df.iloc[start_idx : start_idx + 20]
                    seq = seq[seq['Match ID'] == m_id] 
                    is_c, is_g = False, False
                    start_t = row['timeStamp']
                    for _, nxt in seq.iterrows():
                        dur = (nxt['timeStamp'] - start_t).total_seconds()
                        if dur < 0 or dur > max_d: break
                        if nxt['Team Name'] != t_name and dur > 2: break
                        if nxt['Team Name'] != t_name: continue
                        if int(nxt['typeId']) == 16:
                            is_c, is_g = True, True
                            break
                        if nxt['x'] > 83 or int(nxt['typeId']) in [13, 14, 15]:
                            is_c = True
                            break
                    if is_c:
                        c_count += 1
                        if is_g: g_count += 1
                        origins.append("Presión Alta" if sx > 50 else "Repliegue")

            m_teams = self.df[self.df['Match ID'] == m_id]['Team Name'].unique()
            # El rival es el equipo del Match ID que NO es el equipo actual
            rival_real = [t for t in m_teams if t != t_name]
            rival_nombre = rival_real[0] if rival_real else "Desconocido"

            all_results.append({
                'Equipo': t_name, 
                'Rival': rival_nombre,  # <--- Guardamos el nombre exacto para el buscador
                'Contras': c_count, 
                'Goles': g_count, 
                'Fecha': group['timeStamp'].min()
            })

        self.all_matches_df = pd.DataFrame(all_results)
        self.all_matches_df = self.all_matches_df.sort_values(['Equipo', 'Fecha'])
        self.all_matches_df['Jornada'] = self.all_matches_df.groupby('Equipo').cumcount() + 1
        
        agg = self.all_matches_df.groupby('Equipo').agg({'Contras': ['sum', 'mean'], 'Goles': 'sum'}).reset_index()
        agg.columns = ['Equipo', 'Contras_Totales', 'Contras_Media', 'Goles_Totales']
        agg['Eficiencia'] = agg['Goles_Totales'] / agg['Contras_Totales'].replace(0, 1)
        self.team_stats = agg

    def create_report(self, target_team):
        fig = plt.figure(figsize=(11.69, 8.27), facecolor='white')
        if os.path.exists("assets/fondo_informes.png"):
            ax_bg = fig.add_axes([0, 0, 1, 1], zorder=-1)
            ax_bg.imshow(plt.imread("assets/fondo_informes.png"), aspect='auto', alpha=0.15)
            ax_bg.axis('off')

        gs = fig.add_gridspec(3, 4, height_ratios=[0.12, 0.45, 1], width_ratios=[1.0, 1.0, 1.6, 0.2],
                             hspace=0.6, wspace=0.4, left=0.08, right=0.95, top=0.95, bottom=0.08)

        # 1. TÍTULO
        ax_head = fig.add_subplot(gs[0, :])
        ax_head.axis('off')
        ax_head.text(0.5, 0.5, f'ESTADISTICAS - CONTRAATAQUES', fontsize=24, weight='bold', ha='center', color='#1e3d59')
        
        target_logo = self.find_team_logo_by_similarity(target_team)
        villarreal_logo = self.find_team_logo_by_similarity("Villarreal")
        if target_logo is not None:
            ax_head.add_artist(AnnotationBbox(OffsetImage(target_logo, zoom=0.65), (0.98, 0.5), frameon=False))
        if villarreal_logo is not None:
            ax_head.add_artist(AnnotationBbox(OffsetImage(villarreal_logo, zoom=0.65), (0.92, 0.5), frameon=False))

        # 2. EVOLUTIVO POR JORNADA
        ax_evol = fig.add_subplot(gs[1, :])
        team_evol = self.all_matches_df[self.all_matches_df['Equipo'] == target_team].sort_values('Jornada')

        bars_ev = ax_evol.bar(team_evol['Jornada'], team_evol['Contras'], color='#27ae60', alpha=0.6)

        # Añadir cantidad encima de cada barra
        for bar in bars_ev:
            height = bar.get_height()
            ax_evol.text(bar.get_x() + bar.get_width()/2., height + 0.05, f'{int(height)}', 
                        ha='center', va='bottom', fontsize=10, weight='bold', color='#1e3d59')

        # Espacio extra negativo en Y para que los escudos no se corten
        ax_evol.set_ylim(-1.5, max(team_evol['Contras'].max() + 1.5, 5))

        for _, row in team_evol.iterrows():
            # Buscamos el escudo con la nueva función de similitud de ABP3
            rival_logo = self.find_team_logo_by_similarity(row['Rival'])
            if rival_logo is not None:
                # Coordenada X es la Jornada, Y es -0.8 (debajo del eje)
                imagebox = OffsetImage(rival_logo, zoom=0.1) 
                ab = AnnotationBbox(imagebox, (row['Jornada'], -0.8), frameon=False)
                ax_evol.add_artist(ab)
            
        ax_evol.set_xticks(team_evol['Jornada'])
        ax_evol.set_xticklabels([f"J{j}" for j in team_evol['Jornada']], fontsize=9, weight='bold')

        # 4. BARRAS HORIZONTALES
        # --- Goles ---
        ax_goles = fig.add_subplot(gs[2, 0])
        df_g = self.team_stats.sort_values('Goles_Totales', ascending=True)
        colors_g = ['#2ecc71' if x == target_team else ('#f1c40f' if "villarreal" in x.lower() else '#bdc3c7') for x in df_g['Equipo']]
        ax_goles.barh(range(len(df_g)), df_g['Goles_Totales'], color=colors_g)
        ax_goles.set_yticks(range(len(df_g)))
        ax_goles.set_yticklabels(df_g['Equipo'], fontsize=6)
        ax_goles.set_xticks([])
        ax_goles.set_title("GOLES TOTALES", fontsize=10, weight='bold', pad=10)
        
        for i, (eq, val) in enumerate(zip(df_g['Equipo'], df_g['Goles_Totales'])):
            if eq == target_team or "villarreal" in eq.lower():
                ax_goles.text(val + 0.1, i, f'{int(val)}', va='center', fontsize=7, weight='bold')

        # --- Eficiencia ---
        ax_eff = fig.add_subplot(gs[2, 1])
        df_e = self.team_stats.sort_values('Eficiencia', ascending=True)
        colors_e = ['#2ecc71' if x == target_team else ('#f1c40f' if "villarreal" in x.lower() else '#bdc3c7') for x in df_e['Equipo']]
        ax_eff.barh(range(len(df_e)), df_e['Eficiencia'], color=colors_e)
        ax_eff.set_yticks(range(len(df_e)))
        ax_eff.set_yticklabels(df_e['Equipo'], fontsize=6)
        ax_eff.set_xticks([])
        ax_eff.set_title("GOLES POR CONTRAATAQUE", fontsize=10, weight='bold', pad=10)
        
        for i, (eq, val) in enumerate(zip(df_e['Equipo'], df_e['Eficiencia'])):
            if eq == target_team or "villarreal" in eq.lower():
                ax_eff.text(val + 0.005, i, f'{val:.2f}', va='center', fontsize=7, weight='bold')

        # 5. SCATTER PLOT
        ax_scat = fig.add_subplot(gs[2, 2:])
        ax_scat.set_title("MAPA DE PELIGROSIDAD (VOLUMEN VS EFICIENCIA)", weight='bold', size=11, pad=15)
        x_m, y_m = self.team_stats['Contras_Media'].mean(), self.team_stats['Eficiencia'].mean()
        ax_scat.axvline(x_m, color='#34495e', linestyle='-', lw=1.5, alpha=0.4)
        ax_scat.axhline(y_m, color='#34495e', linestyle='-', lw=1.5, alpha=0.4)

        for _, row in self.team_stats.iterrows():
            es_imp = (row['Equipo'] == target_team or "villarreal" in row['Equipo'].lower())
            logo = self.find_team_logo_by_similarity(row['Equipo'], grayscale=not es_imp)
            if logo is not None:
                ab = AnnotationBbox(OffsetImage(logo, zoom=0.45 if es_imp else 0.22), 
                                    (row['Contras_Media'], row['Eficiencia']), frameon=False)
                ax_scat.add_artist(ab)
            else:
                ax_scat.plot(row['Contras_Media'], row['Eficiencia'], alpha=0)

        x_min, x_max = self.team_stats['Contras_Media'].min(), self.team_stats['Contras_Media'].max()
        y_min, y_max = self.team_stats['Eficiencia'].min(), self.team_stats['Eficiencia'].max()
        ax_scat.set_xlim(x_min * 0.7, x_max * 1.3)
        ax_scat.set_ylim(y_min * 0.7, y_max * 1.3)
        ax_scat.set_xlabel("Media Contraataques / Partido", size=9)
        ax_scat.set_ylabel("Eficiencia (Goles/Contra)", size=9)

        for ax in [ax_evol, ax_goles, ax_eff, ax_scat]:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False if ax != ax_scat else True)
            ax.set_facecolor('none')

        return fig

def main():
    report = AnalizadorContraataquesOpta()
    report.load_data()
    equipos = sorted(report.team_stats['Equipo'].unique())
    for i, eq in enumerate(equipos, 1):
        pass
    idx = int(input(f"\nNúmero (1-{len(equipos)}): ")) - 1
    equipo_sel = equipos[idx]
    fig = report.create_report(equipo_sel)
    output = f"informe_contras_{equipo_sel.replace(' ', '_')}.pdf"
    fig.savefig(output, bbox_inches='tight', dpi=300)
    plt.show()

if __name__ == "__main__":
    main()