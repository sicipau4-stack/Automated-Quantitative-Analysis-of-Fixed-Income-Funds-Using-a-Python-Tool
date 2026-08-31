


# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 15:04:07 2025

@author: O009372
"""


import os
import pandas as pd
import glob
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from scipy.stats import linregress, kurtosis, skew
import warnings
import seaborn as sns
from dateutil.relativedelta import relativedelta
import statsmodels.api as sm 
import subprocess 
import textwrap   
import pmdarima as pm

warnings.filterwarnings('ignore')


# 1. CARGA Y PROCESAMIENTO DE DATOS 
def _procesar_lista_ficheros(ficheros_a_procesar, isins_seleccionados):
    lista_dfs = []
    for file_path in ficheros_a_procesar:
        try:
            df_temp = pd.read_csv(file_path, sep=';', encoding='latin1', on_bad_lines='skip', low_memory=False)
            df_temp.rename(columns={
                'ISIN_SHARE_CLASS': 'REF_ISIN',
                'Duration': 'DURACION',
                'Yield': 'TIR',
                'Fecha': 'DATA_DATE',
                'FUND_NAV_DT': 'fund_nav_dt'
            }, inplace=True)
            if 'REF_ISIN' in df_temp.columns:
                lista_dfs.append(df_temp)
        except Exception as e:
            print(f"Advertencia: No se pudo leer el fichero {file_path}. Error: {e}")
            continue

    if not lista_dfs:
        print("Error: No se pudo cargar ningún dato de esta lista de ficheros.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    df_completo = pd.concat(lista_dfs, ignore_index=True)
    df_filtrado = df_completo[df_completo['REF_ISIN'].isin(isins_seleccionados)].copy()

    if df_filtrado.empty:
        print("Advertencia: Los ISINs seleccionados no se encontraron en este chunk de ficheros.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    for col in ['TIR', 'DURACION', 'precio']:
        if col in df_filtrado.columns:
            df_filtrado[col] = df_filtrado[col].astype(str).str.replace(',', '.').str.replace('%', '').str.strip()
            df_filtrado[col] = pd.to_numeric(df_filtrado[col], errors='coerce')

    df_filtrado['DATA_DATE'] = pd.to_datetime(df_filtrado['DATA_DATE'], dayfirst=True, errors='coerce')
    df_filtrado.dropna(subset=['DATA_DATE', 'REF_ISIN'], inplace=True)
    df_filtrado.sort_values(by=['REF_ISIN', 'DATA_DATE'], inplace=True)
    df_filtrado['fund_nav_dt'] = pd.to_datetime(df_filtrado['fund_nav_dt'], dayfirst=True, errors='coerce')

    df_td_raw = df_filtrado[['REF_ISIN', 'DATA_DATE', 'TIR', 'DURACION']].dropna(how='any')
    df_precio_raw = df_filtrado[['REF_ISIN', 'fund_nav_dt', 'precio']].dropna(how='any')

    lista_dfs_procesados = []
    columnas_estaticas = ['category_region', 'nav_crncy'] 

    for isin, group in df_filtrado.groupby('REF_ISIN'):
        df_td = group[['DATA_DATE', 'TIR', 'DURACION']].dropna(subset=['DATA_DATE']).set_index('DATA_DATE').sort_index()
        df_td_resampled = df_td.resample('M').mean()
        
        df_precio = group[['fund_nav_dt', 'precio']].dropna(subset=['fund_nav_dt', 'precio']).set_index('fund_nav_dt').sort_index()
        df_precio_resampled = df_precio.resample('M').last()

        df_merged = pd.concat([df_td_resampled, df_precio_resampled], axis=1)
        df_interpolated = df_merged.interpolate(method='linear', limit_direction='both')
        df_interpolated['REF_ISIN'] = isin

        for col in columnas_estaticas:
            if col in group.columns and not group[col].isnull().all():
                static_value = group[col].bfill().ffill().iloc[0]
                df_interpolated[col] = static_value
            else:
                df_interpolated[col] = np.nan
        
        lista_dfs_procesados.append(df_interpolated)

    if not lista_dfs_procesados:
        print("Error: No se pudieron procesar datos para el análisis en este chunk.")
        return pd.DataFrame(), pd.DataFrame(), df_td_raw, df_precio_raw

    df_ancho = pd.concat(lista_dfs_procesados).reset_index()
    df_ancho.rename(columns={'index': 'DATA_DATE'}, inplace=True)
    
    df_ancho.dropna(subset=['TIR', 'DURACION'], how='any', inplace=True)
    
    df_largo = pd.melt(df_ancho,
                        id_vars=['REF_ISIN', 'DATA_DATE'],
                        value_vars=['TIR', 'DURACION'],
                        var_name='DATAFIELD',
                        value_name='FUND_VALUE')
    
    return df_ancho, df_largo, df_td_raw, df_precio_raw


def cargar_datos_en_chunks(ruta_carpeta, mascara_ficheros, isins_seleccionados):
    ruta_completa = os.path.join(ruta_carpeta, mascara_ficheros)
    ficheros_todos = sorted(glob.glob(ruta_completa))
    
    ficheros_antiguos = []
    ficheros_nuevos = []
    fecha_corte = pd.to_datetime('2024-10-31') # los chunks se han añadido porque no me funciona la lectura completa de ficheros, por lo que se divide en dos partes más pequeñas (chunks) y si los lee

    for f in ficheros_todos:
        try:
            nombre_base = os.path.basename(f)
            fecha_str = nombre_base.split('_')[0]
            fecha_fichero = pd.to_datetime(fecha_str + '01', format='%Y%m%d') 
            if fecha_fichero <= fecha_corte:
                ficheros_antiguos.append(f)
            else:
                ficheros_nuevos.append(f)
        except Exception:
            print(f"Ignorando fichero con nombre no estándar: {f}")
            

    # Procesar chunk 1
    ancho1, largo1, td_raw1, precio_raw1 = _procesar_lista_ficheros(ficheros_antiguos, isins_seleccionados)
    
    # Procesar chunk 2
    ancho2, largo2, td_raw2, precio_raw2 = _procesar_lista_ficheros(ficheros_nuevos, isins_seleccionados)

    # Combinar los cuatro dataframes finales 
    df_ancho_final = pd.concat([ancho1, ancho2]) \
                       .sort_values(by='DATA_DATE') \
                       .drop_duplicates(subset=['REF_ISIN', 'DATA_DATE'], keep='last') \
                       .sort_values(by=['REF_ISIN', 'DATA_DATE']).reset_index(drop=True)
                       
    df_largo_final = pd.concat([largo1, largo2]) \
                       .sort_values(by='DATA_DATE') \
                       .drop_duplicates(subset=['REF_ISIN', 'DATA_DATE', 'DATAFIELD'], keep='last') \
                       .sort_values(by=['REF_ISIN', 'DATA_DATE']).reset_index(drop=True)
    
    df_td_raw_final = pd.concat([td_raw1, td_raw2]).sort_values(by=['REF_ISIN', 'DATA_DATE']).reset_index(drop=True)
    df_precio_raw_final = pd.concat([precio_raw1, precio_raw2]).sort_values(by=['REF_ISIN', 'fund_nav_dt']).reset_index(drop=True)
    
    return df_ancho_final, df_largo_final, df_td_raw_final, df_precio_raw_final



# 2. SELECCIÓN DE ISINs Y GUARDADO 
def guardar_historicos_separados(df_ancho, df_precio_raw, isins_seleccionados, ruta_salida):
    print("\nGuardando ficheros CSV históricos (TIR/duración y precio) separados por ISIN...")
    if not os.path.exists(ruta_salida):
        os.makedirs(ruta_salida)

    for isin in isins_seleccionados:
        # Histórico de TIR y duración
        df_td_isin = df_ancho[df_ancho['REF_ISIN'] == isin].copy()
        if not df_td_isin.empty:
            columnas_td = ['DATA_DATE', 'REF_ISIN', 'TIR', 'DURACION']
            columnas_td = [c for c in columnas_td if c in df_td_isin.columns]
            df_td_isin = df_td_isin[columnas_td].dropna(subset=['TIR', 'DURACION'])
            df_td_isin.sort_values(by='DATA_DATE', inplace=True)
            for col in ('TIR', 'DURACION'):
                df_td_isin[col] = pd.to_numeric(df_td_isin[col], errors='coerce')
                if not df_td_isin[col].empty and df_td_isin[col].median() > 50:
                    print(f"  Corrigiendo escala de {col} para {isin} (división entre 1000)")
                    df_td_isin[col] = df_td_isin[col] / 1000
            nombre_archivo_td = f"{isin}_historia_tir_duracion.csv"
            ruta_archivo_td = os.path.join(ruta_salida, nombre_archivo_td)
            df_td_isin.to_csv(ruta_archivo_td, index=False, sep=';', decimal='.', float_format='%.2f')
            print(f"+ Archivo de TIR/duración generado para {isin}: {ruta_archivo_td}")
        else:
            print(f" No hay datos de TIR/duración para {isin}")

        # Histórico de precio
        df_precio_isin = df_ancho[df_ancho['REF_ISIN'] == isin].copy()
        
        if not df_precio_isin.empty:
            columnas_p = ['DATA_DATE', 'REF_ISIN', 'precio']
            columnas_p_existentes = [c for c in columnas_p if c in df_precio_isin.columns]
            
            if 'precio' in columnas_p_existentes:
                df_precio_isin_limpio = df_precio_isin[columnas_p_existentes].dropna(subset=['precio'])
            else:
                df_precio_isin_limpio = pd.DataFrame(columns=columnas_p_existentes) 
            
            if not df_precio_isin_limpio.empty:
                df_precio_isin_limpio.sort_values(by='DATA_DATE', inplace=True)
                nombre_archivo_p = f"{isin}_historia_precio.csv"
                ruta_archivo_p = os.path.join(ruta_salida, nombre_archivo_p)
                
                df_precio_isin_limpio.to_csv(ruta_archivo_p, index=False, sep=';', decimal='.', float_format='%.2f')
                print(f"+ Archivo de precio generado para {isin}: {ruta_archivo_p}")
            else:
                print(f" No hay datos de precio (en df_ancho) para {isin}")
        else:
            print(f" No hay datos en df_ancho para {isin} (para precios)")
       
            
def seleccionar_origen_isins(df_disponible):
    ruta_fichero_isin = r"C:\Users\Usuario\Desktop\Tfg Economía\input\isins hard currency.xlsx"
    if df_disponible is None or df_disponible.empty: return []
    isins_disponibles = sorted(df_disponible['REF_ISIN'].dropna().unique().tolist())
    if not isins_disponibles: return []
    while True:
        print("\n ¿Cómo quieres seleccionar los ISINs?")
        print(f"1. Desde el fichero Excel ({ruta_fichero_isin})")
        print("2. Introducirlos manualmente")
        opcion = input("Elige una opción (1 o 2): ")
        if opcion == '1':
            try:
                df_excel = pd.read_excel(ruta_fichero_isin)
                if 'ISIN' not in df_excel.columns:
                    print(f"\nError: El fichero Excel no contiene la columna requerida 'ISIN'.")
                    continue
                isins_del_fichero = [str(isin).strip() for isin in df_excel['ISIN'].dropna().tolist()]
                if not isins_del_fichero: continue
                isins_validos = [isin for isin in isins_del_fichero if isin in isins_disponibles]
                isins_invalidos = [isin for isin in isins_del_fichero if isin not in isins_disponibles]
                if isins_invalidos:
                    print(f"\nAdvertencia: Los siguientes ISINs del fichero no se encontraron y serán ignorados: {', '.join(isins_invalidos)}")
                if not isins_validos:
                    print("\nError: Ninguno de los ISINs del fichero se encontró en los datos cargados.")
                    continue
                print(f"\nISINs cargados y validados desde el fichero: {', '.join(isins_validos)}")
                return isins_validos
            except FileNotFoundError:
                print(f"\nError: No se pudo encontrar el fichero en la ruta: '{ruta_fichero_isin}'")
                continue
            except Exception as e:
                print(f"\nOcurrió un error inesperado al leer el fichero Excel: {e}")
                continue
        elif opcion == '2':
            print("\n Selección manual de ISIN ")
            print("ISINs disponibles en los datos:")
            for isin in isins_disponibles:
                print(f"- {isin}")
            while True:
                print("\nPor favor, copia y pega los ISINs que quieras analizar, separados por comas:")
                entrada_usuario = input("> ")
                isins_seleccionados = [isin.strip() for isin in entrada_usuario.split(',') if isin.strip()]
                isins_invalidos_manual = [isin for isin in isins_seleccionados if isin not in isins_disponibles]
                if not isins_seleccionados:
                    print("No has introducido ningún ISIN. Por favor, inténtalo de nuevo.")
                elif not isins_invalidos_manual:
                    print(f"Has seleccionado: {', '.join(isins_seleccionados)}")
                    return isins_seleccionados
                else:
                    print(f"\nError: Los siguientes ISINs no son válidos o no se encontraron: {', '.join(isins_invalidos_manual)}")
        else:
            print("Opción no válida. Por favor, introduce '1' o '2'.")



# 3. FUNCIONES DE GRAFICACIÓN 
def graficar_precios_y_retornos(df_ancho, isins_seleccionados, ruta_salida, columna_precio='precio'):
    try:
        if columna_precio not in df_ancho.columns or df_ancho[columna_precio].isnull().all():
            print(f"  ADVERTENCIA: No se puede generar el gráfico porque la columna '{columna_precio}' está vacía o no existe en df_ancho.")
            return None
        df_plot = df_ancho.copy()
        
        df_plot['retorno'] = df_plot.groupby('REF_ISIN')[columna_precio].pct_change() * 100 

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(17, 14), sharex=True)
        fig.suptitle(f'Evolución de "{columna_precio.capitalize()}" y sus retornos mensuales', fontsize=20, fontweight='bold')
        ax1.set_title(f'Evolución mensual de {columna_precio.capitalize()}', fontsize=16)
        for isin in isins_seleccionados:
            df_isin = df_plot[df_plot['REF_ISIN'] == isin].sort_values('DATA_DATE')
            if not df_isin.empty:
                ax1.plot(df_isin['DATA_DATE'], df_isin[columna_precio], marker='o', linestyle='-', label=isin)
        ax1.set_ylabel('Precio')
        ax1.grid(True, which='major', linestyle='--', linewidth=0.5)
        ax1.legend(title="ISINs", bbox_to_anchor=(1.02, 1), loc='upper left')
        ax2.set_title('Retornos mensuales calculados', fontsize=16)
        for isin in isins_seleccionados:
            df_isin = df_plot[df_plot['REF_ISIN'] == isin].sort_values('DATA_DATE')
            if not df_isin.empty:
                ax2.plot(df_isin['DATA_DATE'], df_isin['retorno'], marker='.', linestyle='--', label=isin)
        ax2.set_ylabel('Retorno mensual (%)')
        ax2.set_xlabel('Fecha')
        ax2.grid(True, which='major', linestyle='--', linewidth=0.5)
        ax2.legend(title="ISINs", bbox_to_anchor=(1.02, 1), loc='upper left')
        ax2.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
        fig.autofmt_xdate()
        fig.tight_layout(rect=[0, 0, 0.9, 0.96])
        nombre_fichero_grafico = f"precios_y_retornos_{len(isins_seleccionados)}_isins.png"
        ruta_grafico = os.path.join(ruta_salida, nombre_fichero_grafico)
        plt.savefig(ruta_grafico, dpi=300, bbox_inches='tight')
        plt.close()
        return ruta_grafico
    except Exception as e:
        print(f"  ERROR al generar 'graficar_precios_y_retornos': {e}")
        return None

def graficar_evolucion_historica(df_largo, isins_seleccionados, ruta_salida):
    try:
        df_tir = df_largo[df_largo['DATAFIELD'] == 'TIR'].copy()
        df_duracion = df_largo[df_largo['DATAFIELD'] == 'DURACION']
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(17, 14), sharex=True)
        fig.suptitle('Comparativa de evolución de TIR y duración', fontsize=20, fontweight='bold')
        ax1.set_title('Evolución de la TIR', fontsize=16)
        for isin in isins_seleccionados:
            df_isin_tir = df_tir[df_tir['REF_ISIN'] == isin].sort_values('DATA_DATE')
            if not df_isin_tir.empty: ax1.plot(df_isin_tir['DATA_DATE'], df_isin_tir['FUND_VALUE'], marker='o', linestyle='-', label=isin)
        ax1.set_ylabel('TIR')
        ax1.grid(True, which='major', linestyle='--', linewidth=0.5)
        ax1.legend(title="ISINs", bbox_to_anchor=(1.02, 1), loc='upper left')
        ax2.set_title('Evolución de la duración', fontsize=16)
        for isin in isins_seleccionados:
            df_isin_dur = df_duracion[df_duracion['REF_ISIN'] == isin].sort_values('DATA_DATE')
            if not df_isin_dur.empty: ax2.plot(df_isin_dur['DATA_DATE'], df_isin_dur['FUND_VALUE'], marker='.', linestyle='--', label=isin)
        ax2.set_ylabel('Duración')
        ax2.set_xlabel('Fecha')
        ax2.grid(True, which='major', linestyle='--', linewidth=0.5)
        ax2.legend(title="ISINs", bbox_to_anchor=(1.02, 1), loc='upper left')
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
        fig.autofmt_xdate()
        fig.tight_layout(rect=[0, 0, 0.9, 0.96])
        nombre_fichero_grafico = f"evolucion_historica_{len(isins_seleccionados)}_isins.png"
        ruta_grafico = os.path.join(ruta_salida, nombre_fichero_grafico)
        plt.savefig(ruta_grafico, dpi=300, bbox_inches='tight')
        plt.close()
        return ruta_grafico
    except Exception as e:
        print(f"  ERROR al generar 'evolucion_historica': {e}")
        return None

def graficar_dispersion_tir_duracion(df_largo, ruta_salida):
    try:
        df_tir = df_largo[df_largo['DATAFIELD'] == 'TIR'].copy()
        df_duracion = df_largo[df_largo['DATAFIELD'] == 'DURACION']
        promedios_tir = df_tir.groupby('REF_ISIN')['FUND_VALUE'].mean()
        promedios_duracion = df_duracion.groupby('REF_ISIN')['FUND_VALUE'].mean()
        df_promedios = pd.DataFrame({'TIR_Promedio': promedios_tir, 'Duracion_Promedio': promedios_duracion}).reset_index()
        if df_promedios.empty: return None
        media_total_tir = df_promedios['TIR_Promedio'].mean()
        media_total_duracion = df_promedios['Duracion_Promedio'].mean()
        fig, ax = plt.subplots(figsize=(12, 8))
        for index, row in df_promedios.iterrows():
            ax.scatter(row['Duracion_Promedio'], row['TIR_Promedio'], s=100, label=row['REF_ISIN'], alpha=0.7)
            ax.text(row['Duracion_Promedio'], row['TIR_Promedio'] + 0.0005, row['REF_ISIN'], fontsize=9)
        ax.scatter(media_total_duracion, media_total_tir, s=250, color='red', marker='*', edgecolor='black', label='Media del grupo')
        ax.axhline(y=media_total_tir, color='grey', linestyle='--', linewidth=0.8)
        ax.axvline(x=media_total_duracion, color='grey', linestyle='--', linewidth=0.8)
        ax.set_title('Dispersión de activos por TIR y duración promedio', fontsize=16, fontweight='bold')
        ax.set_xlabel('Duración promedio')
        ax.set_ylabel('TIR promedio')
        ax.legend(title="Referencia")
        ax.grid(True, which='major', linestyle='--', linewidth=0.5)
        fig.tight_layout()
        ruta_grafico = os.path.join(ruta_salida, "dispersion_tir_duracion.png")
        plt.savefig(ruta_grafico, dpi=300, bbox_inches='tight')
        plt.close()
        return ruta_grafico
    except Exception as e:
        print(f"  ERROR al generar 'dispersion_tir_duracion': {e}")
        return None

def graficar_boxplot_comparativo(df_largo, ruta_salida):
    try:
        df_tir = df_largo[df_largo['DATAFIELD'] == 'TIR'].copy()
        df_duracion = df_largo[df_largo['DATAFIELD'] == 'DURACION']
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        fig.suptitle('Comparación de distribuciones por ISIN', fontsize=20, fontweight='bold')
        sns.boxplot(x='REF_ISIN', y='FUND_VALUE', data=df_tir, ax=ax1, palette='viridis')
        ax1.set_title('Distribución de TIR', fontsize=16)
        ax1.set_xlabel('ISIN'); ax1.set_ylabel('TIR'); ax1.tick_params(axis='x', rotation=45)
        sns.boxplot(x='REF_ISIN', y='FUND_VALUE', data=df_duracion, ax=ax2, palette='plasma')
        ax2.set_title('Distribución de duración', fontsize=16)
        ax2.set_xlabel('ISIN'); ax2.set_ylabel('Duración'); ax2.tick_params(axis='x', rotation=45)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        ruta_grafico = os.path.join(ruta_salida, "comparacion_boxplot.png")
        plt.savefig(ruta_grafico, dpi=300, bbox_inches='tight')
        plt.close()
        return ruta_grafico
    except Exception as e:
        print(f"  ERROR al generar 'boxplot_comparativo': {e}")
        return None


def graficar_heatmap_correlacion(df_ancho, isins_seleccionados, ruta_salida):
    """
    Genera un heatmap de la correlación de retornos mensuales.
    """
    try:
        if 'precio' not in df_ancho.columns or df_ancho['precio'].isnull().all():
            print("  ADVERTENCIA: No hay datos de 'precio' para calcular heatmap de correlación.")
            return None
        if len(isins_seleccionados) <= 1:
            print("  ADVERTENCIA: Se necesita más de un ISIN para heatmap de correlación.")
            return None
            
        # Calcular retornos mensuales
        df_retornos = df_ancho[['DATA_DATE', 'REF_ISIN', 'precio']].copy()
        df_retornos.sort_values(by=['REF_ISIN', 'DATA_DATE'], inplace=True)
        df_retornos['retorno'] = df_retornos.groupby('REF_ISIN')['precio'].pct_change() * 100
        df_pivot_ret = df_retornos.pivot_table(index='DATA_DATE', columns='REF_ISIN', values='retorno')
        df_pivot_ret.dropna(axis=1, how='all', inplace=True) 
        df_pivot_ret.dropna(axis=0, how='any', inplace=True) 

        if df_pivot_ret.shape[1] > 1:
            corr_matrix_ret = df_pivot_ret.corr()
            
            plt.figure(figsize=(12, 10))
            sns.heatmap(corr_matrix_ret, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, vmin=-1, vmax=1)
            plt.title('Mapa de calor de correlación de retornos mensuales', fontsize=16, fontweight='bold')
            plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0)
            plt.tight_layout()
            ruta_grafico = os.path.join(ruta_salida, "heatmap_correlacion_retornos.png")
            plt.savefig(ruta_grafico, dpi=300, bbox_inches='tight')
            plt.close()
            return ruta_grafico
        else:
            print("  ADVERTENCIA: Datos insuficientes para heatmap de correlación tras limpiar NaNs.")
            return None
            
    except Exception as e_heat:
        print(f"  ERROR al generar 'graficar_heatmap_correlacion': {e_heat}")
        return None

def graficar_volatilidad_movil_tir(df_largo, isins_seleccionados, ruta_salida):
    try:
        df_tir = df_largo[df_largo['DATAFIELD'] == 'TIR'].copy()
        if df_tir.empty:
            print("    - No hay datos de TIR para calcular volatilidad móvil.")
            return None

        fig, ax = plt.subplots(figsize=(17, 8))
        ax.set_title('Volatilidad de la TIR (Desv. est. móvil - ventana 12 meses)', fontsize=16, fontweight='bold')
        ventana = 12 
        count_valid = 0
        for isin in isins_seleccionados:
            datos_isin = df_tir[df_tir['REF_ISIN'] == isin].set_index('DATA_DATE').sort_index()['FUND_VALUE']
            datos_isin = datos_isin[~datos_isin.index.duplicated(keep='last')]
            if len(datos_isin) > ventana:
                volatilidad = datos_isin.rolling(window=ventana).std()
                ax.plot(volatilidad.index, volatilidad.values, label=f'{isin}', marker='.', markersize=4)
                count_valid += 1
            else:
                print(f"    - No hay suficientes datos de TIR ({len(datos_isin)}) para ventana de {ventana} para {isin}")

        if count_valid == 0:
            plt.close(fig)
            print("    - No se generó gráfico de volatilidad de TIR (datos insuficientes).")
            return None

        ax.set_xlabel('Fecha', fontsize=12)
        ax.set_ylabel('Volatilidad móvil mensual (p.p.)', fontsize=12)
        ax.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.legend(title="ISINs", bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        fig.autofmt_xdate()
        fig.tight_layout()
        ruta_grafico = os.path.join(ruta_salida, "volatilidad_movil_tir.png")
        plt.savefig(ruta_grafico, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return ruta_grafico
    except Exception as e:
        print(f"  ERROR al generar 'graficar_volatilidad_movil_tir': {e}")
        try: plt.close(fig)
        except: pass
        return None

def graficar_volatilidad_movil_retornos(df_ancho, isins_seleccionados, ruta_salida):
    try:
        if 'precio' not in df_ancho.columns or df_ancho['precio'].isnull().all():
            print("    - No hay datos de 'precio' para calcular volatilidad de retornos.")
            return None

        df_ret_vol = df_ancho[['DATA_DATE', 'REF_ISIN', 'precio']].copy()
        df_ret_vol.sort_values(by=['REF_ISIN', 'DATA_DATE'], inplace=True)
        df_ret_vol['retorno'] = df_ret_vol.groupby('REF_ISIN')['precio'].pct_change() * 100

        ventana_vol = 12 
        fig, ax = plt.subplots(figsize=(17, 8))
        ax.set_title('Volatilidad de retornos mensuales (desv. est. móvil - ventana 12 meses)', fontsize=16, fontweight='bold')
        count_valid_vol = 0

        for isin in isins_seleccionados:
            datos_isin_ret = df_ret_vol[df_ret_vol['REF_ISIN'] == isin].set_index('DATA_DATE').sort_index()['retorno']
            datos_isin_ret = datos_isin_ret[~datos_isin_ret.index.duplicated(keep='last')].dropna()
            
            if len(datos_isin_ret) > ventana_vol:
                volatilidad_ret = datos_isin_ret.rolling(window=ventana_vol).std()
                ax.plot(volatilidad_ret.index, volatilidad_ret.values, label=f'{isin}', marker='.', markersize=4)
                count_valid_vol += 1
            else:
                print(f"    - No hay suficientes datos de retorno ({len(datos_isin_ret)}) para ventana de {ventana_vol} para {isin}")

        if count_valid_vol == 0:
            plt.close(fig)
            print("    - No se generó gráfico de volatilidad de retornos (datos insuficientes).")
            return None

        ax.set_xlabel('Fecha', fontsize=12)
        ax.set_ylabel('Volatilidad Móvil Mensual (%)', fontsize=12)
        ax.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.legend(title="ISINs", bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1f}%'.format(y)))
        fig.autofmt_xdate()
        fig.tight_layout()
        ruta_grafico = os.path.join(ruta_salida, "volatilidad_movil_retornos.png")
        plt.savefig(ruta_grafico, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return ruta_grafico

    except Exception as e_vol:
        print(f"  ERROR al generar 'graficar_volatilidad_movil_retornos': {e_vol}")
        try: plt.close(fig) 
        except: pass
        return None    



def calcular_ft_adapted_volatility(retornos_series):
    """
    Calcula la Ft adapted volatility:
    1. Innovaciones: residuos autoarima (d=0, max_p=2, max_q=1).
    2. Phi: frecuencia de residuos estandarizados > 1.96.
    3. Varianza: sumatorio de 36 meses.
    """
    # 1. Autoarima
    try:
        model = pm.auto_arima(retornos_series,
                              start_p=0, start_q=0,
                              max_p=2, max_q=1,
                              d=0,
                              stationary=True,
                              seasonal=False,
                              information_criterion='aic',
                              suppress_warnings=True,
                              error_action='ignore')
        a_t = pd.Series(model.resid(), index=retornos_series.index)
        # Guardamos el orden del modelo (p,d,q) para el informe
        modelo_str = f"ARIMA{model.order}"
    except Exception as e:
        print(f"Advertencia: Fallo en autoarima ({e}), usando media simple.")
        a_t = retornos_series - retornos_series.mean()
        

    # 2. Cálculo de phi
    media_residuos = a_t.mean()
    sigma_residuos = a_t.std()
    
    if sigma_residuos == 0:
        return pd.Series(0, index=retornos_series.index), 0.01

    residuos_estandarizados = np.abs((a_t - media_residuos) / sigma_residuos)
    umbral = 1.96
    shocks = residuos_estandarizados > umbral
    u = shocks.sum()
    n = len(retornos_series)
    phi = u / n
    if phi == 0: phi = 0.01 
        
    # 3. Varianza (ventana de 36 meses)
    vol_values = []
    a_values = a_t.values
    decay_factor = 1 - phi
    WINDOW_SIZE = 36 
    
    for t in range(n):
        inicio_ventana = max(0, t - WINDOW_SIZE + 1)
        errores_ventana = a_values[inicio_ventana : t+1][::-1] 
        indices = np.arange(len(errores_ventana))
        pesos = phi * (decay_factor ** indices)
        varianza_t = np.sum(pesos * (errores_ventana**2))
        vol_values.append(np.sqrt(varianza_t))
        
    # 4. Anualización
    vol_ft_series = pd.Series(vol_values, index=retornos_series.index) * np.sqrt(12)
    
    return vol_ft_series, phi, modelo_str

def graficar_ft_adapted_volatility(df_ancho, isins_seleccionados, ruta_salida):
    try:
        if 'precio' not in df_ancho.columns or df_ancho['precio'].isnull().all():
            print("    - No hay datos de 'precio' para Ft adapted volatility.")
            return None, {}, {}

        cols_necesarias = ['DATA_DATE', 'REF_ISIN', 'precio']
        if 'Indice_Referencia' in df_ancho.columns:
            cols_necesarias.append('Indice_Referencia')
            
        df_ret = df_ancho[cols_necesarias].copy()
        df_ret.sort_values(by=['REF_ISIN', 'DATA_DATE'], inplace=True)
        

        df_ret['retorno'] = df_ret.groupby('REF_ISIN')['precio'].pct_change()*100
        
        fig, ax = plt.subplots(figsize=(17, 8))
        ax.set_title('Ft adapted volatility', fontsize=16, fontweight='bold')
        
        phis_calculados = {}
        arimas_calculados = {} 
        count_valid = 0
        

        FECHA_INICIO_PLOT = pd.to_datetime('2021-03-01')
        FECHA_FIN_PLOT = pd.to_datetime('2025-09-30')


        if 'Indice_Referencia' in df_ret.columns:
            indices_usados = df_ret[df_ret['REF_ISIN'].isin(isins_seleccionados)]['Indice_Referencia'].dropna().unique()
            
            dic_indices = {
                 "Bloomberg Euro Agg Bond TR EUR": "Bloomberg Euro Agg Bond TR EUR",
                 "Bloomberg US Agg Bond TR USD": "Bloomberg US Agg Bond TR USD",
                 "Bloomberg Global Aggregate TR Hdg USD": "Bloomberg Global Aggregate TR Hdg USD",
                 "Bloomberg Global Aggregate TR Hdg EUR": "Bloomberg Global Aggregate TR Hdg EUR",
            }

            RUTA_INDICES = r"C:\Users\Usuario\Desktop\Tfg Economía\Rentab Indices RF.xlsx"

            if len(indices_usados) == 1 and indices_usados[0] in dic_indices:
                nombre_indice_informe = indices_usados[0]
                columna_indice_excel = dic_indices[nombre_indice_informe]
                
                try:
                    if os.path.exists(RUTA_INDICES):
                        df_ind = pd.read_excel(RUTA_INDICES)
                        df_ind.rename(columns={df_ind.columns[0]: "Fecha"}, inplace=True)
                        df_ind["Fecha"] = pd.to_datetime(df_ind["Fecha"], errors='coerce')
                        
                        if columna_indice_excel in df_ind.columns:
                            df_ind.sort_values('Fecha', inplace=True)
                            df_ind.set_index('Fecha', inplace=True)
                            
                            raw_data = df_ind[columna_indice_excel].dropna()
                            
                            retorno_indice = raw_data 
                          
                            if len(retorno_indice) > 36:
                                vol_ft_idx, phi_idx, _ = calcular_ft_adapted_volatility(retorno_indice)
                                
                                vol_plot_idx = vol_ft_idx[(vol_ft_idx.index >= FECHA_INICIO_PLOT) & (vol_ft_idx.index <= FECHA_FIN_PLOT)]
                                
                                if not vol_plot_idx.empty:
                                    ax.plot(vol_plot_idx.index, vol_plot_idx.values, 
                                            label=f'BENCHMARK: {nombre_indice_informe}', 
                                            color='black', linestyle='--', linewidth=2, alpha=0.8)

                except Exception as e_ind:
                    print(f"    - No se pudo añadir el índice al gráfico: {e_ind}")


        for isin in isins_seleccionados:
            datos_isin = df_ret[df_ret['REF_ISIN'] == isin].set_index('DATA_DATE')['retorno'].dropna()
            
            if len(datos_isin) > 36:
                vol_ft, phi, modelo_str = calcular_ft_adapted_volatility(datos_isin)
                
                phis_calculados[isin] = phi
                arimas_calculados[isin] = modelo_str # Guardamos el modelo
                
                vol_plot = vol_ft[(vol_ft.index >= FECHA_INICIO_PLOT) & (vol_ft.index <= FECHA_FIN_PLOT)]
                
                if not vol_plot.empty:
                    ax.plot(vol_plot.index, vol_plot.values, label=f'{isin}', linewidth=1.5)
                    count_valid += 1
            else:
                print(f"    - Datos insuficientes para autoarima en {isin}")

        if count_valid == 0:
            plt.close(fig)
            return None, {}, {}

        ax.set_ylabel('Volatilidad anualizada (%)')
        ax.set_xlabel('Fecha')
        ax.grid(True, which='major', linestyle='--', linewidth=0.5)
        # Leyenda centrada abajo
        ax.legend(title="Activo", bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.set_xlim(left=FECHA_INICIO_PLOT, right=min(FECHA_FIN_PLOT, df_ret['DATA_DATE'].max()))
        
        fig.tight_layout()
        
        ruta_grafico = os.path.join(ruta_salida, "ft_adapted_volatility.png")
        plt.savefig(ruta_grafico, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        return ruta_grafico, phis_calculados, arimas_calculados

    except Exception as e:
        print(f"  ERROR en graficar_ft_adapted_volatility: {e}")
        import traceback
        traceback.print_exc()
        return None, {}, {}

def graficar_distribucion_cambios_mensuales(df_largo, isins_seleccionados, ruta_salida):
    try:
        df_tir = df_largo[df_largo['DATAFIELD'] == 'TIR']
        plt.figure(figsize=(14, 8))
        plt.title('Distribución de cambios mensuales de la TIR (estimación KDE)', fontsize=16, fontweight='bold')
        for isin in isins_seleccionados:
            datos_isin = df_tir[df_tir['REF_ISIN'] == isin].set_index('DATA_DATE').sort_index()
            cambios_mensuales = datos_isin['FUND_VALUE'].diff().dropna()
            if not cambios_mensuales.empty:
                sns.kdeplot(cambios_mensuales, label=isin, fill=True, alpha=0.2)
        plt.xlabel('Cambio mensual en la TIR'); plt.ylabel('Densidad')
        plt.grid(True, which='major', linestyle='--', linewidth=0.5); plt.legend(title="ISINs")
        ruta_grafico = os.path.join(ruta_salida, "distribucion_cambios_mensuales.png")
        plt.savefig(ruta_grafico, dpi=300, bbox_inches='tight')
        plt.close()
        return ruta_grafico
    except Exception as e:
        print(f"  ERROR al generar 'distribucion_cambios_mensuales': {e}")
        return None


# 4. FUNCIÓN PARA LLAMAR A PANDOC
def llamar_pandoc(ruta_md, ruta_salida_principal):
    print("\nIniciando conversión con Pandoc...")
    
    # 1. Generar HTML
    try:
        archivo_md_basename = os.path.basename(ruta_md) 
        archivo_html = "informe_analitico.html"
        ruta_html_salida = os.path.join(ruta_salida_principal, archivo_html)
        
        
        comando_html = [
            "pandoc",
            archivo_md_basename,
            "--standalone",         
            "--mathjax",
            "--toc",
            "--number-sections",    
            "-o", archivo_html      
        ]
        
        print(f"Ejecutando comando: {' '.join(comando_html)}")
        

        subprocess.run(comando_html, check=True, cwd=ruta_salida_principal)
        
        print(f"+ Informe HTML generado con éxito: {ruta_html_salida}")
        
    except FileNotFoundError:
        print("  No se ha podido generar el archivo HTML.")
    except Exception as e:
        print(f"\n  ERROR al generar HTML con Pandoc: {e}")

# 2. Generar PDF
    try:
        archivo_md_basename = os.path.basename(ruta_md)
        archivo_pdf = "informe_analitico.pdf"
        ruta_pdf_salida = os.path.join(ruta_salida_principal, archivo_pdf)

    
        comando_pdf = [
            "pandoc",
            archivo_md_basename,
            "--pdf-engine=C:\\Users\\Usuario\\AppData\\Local\\Programs\\MiKTeX\\miktex\\bin\\x64\\xelatex.exe",
            "--toc",
            "--number-sections",     
            "-o", archivo_pdf
            ]
    

        print(f"Ejecutando comando PDF: {' '.join(comando_pdf)}")
        resultado = subprocess.run(comando_pdf, cwd=ruta_salida_principal, capture_output=True, text=True)

        if resultado.returncode == 0:
            print(f"+ Informe PDF generado con éxito: {ruta_pdf_salida}")
        else:
            print("Error al generar PDF:")
            print(resultado.stdout)
            print(resultado.stderr)
    except Exception as e:
        print(f"ERROR al intentar generar el PDF: {e}")

# 5. FUNCIÓN PRINCIPAL DE ANÁLISIS 
def generar_informe_analitico(df_largo, df_ancho, isins_seleccionados, df_resultados_comparativa, ruta_salida, ruta_salida_fig):
    """
    MODIFICADO:
    - Acepta 'ruta_salida_fig' para guardar gráficos.
    - Genera un 'informe_analitico.md' con cabecera Pandoc.
    - Usa formato de imagen Markdown: ![...](fig/...)
    - Devuelve la ruta al .md generado.
    """
    
    print("\n Iniciando generación de gráficos e informe (.md)...")
    if df_largo.empty:
        print("Error: No hay datos analizables para generar el informe.")
        return None

    print("1. Generando gráfico de evolución histórica...")
    ruta_evolucion = graficar_evolucion_historica(df_largo, isins_seleccionados, ruta_salida_fig)
    
    print("2. Generando gráfico de dispersión...")
    ruta_dispersion = graficar_dispersion_tir_duracion(df_largo, ruta_salida_fig)
    
    print("3. Generando diagramas de caja...")
    ruta_boxplot = graficar_boxplot_comparativo(df_largo, ruta_salida_fig)
    
    print("4. Generando mapa de calor de correlación...")
    ruta_heatmap_ret = graficar_heatmap_correlacion(df_ancho, isins_seleccionados, ruta_salida_fig)
    
    print("5. Generando gráfico de volatilidad de la TIR...")
    ruta_volatilidad_tir = graficar_volatilidad_movil_tir(df_largo, isins_seleccionados, ruta_salida_fig)
    
    print("6. Generando gráfico de volatilidad de los retornos...")
    ruta_volatilidad_retornos = graficar_volatilidad_movil_retornos(df_ancho, isins_seleccionados, ruta_salida_fig)
    
    print("7. Generando distribución de cambios mensuales...")
    ruta_cambios = graficar_distribucion_cambios_mensuales(df_largo, isins_seleccionados, ruta_salida_fig)
    
    print("8. Generando gráficos de retorno vs índice...")
    
    print("9. Generando gráfico de evolución histórica del precio y retornos...")
    # Este gráfico es ADICIONAL, lo metemos también en la carpeta 'fig'
    ruta_precios_retornos = graficar_precios_y_retornos(df_ancho, isins_seleccionados, ruta_salida_fig, columna_precio='precio')
    
    print("10. Generando gráfico de Ft adapted volatility...")
    ruta_ft_vol, dict_phis, dict_arimas = graficar_ft_adapted_volatility(df_ancho, isins_seleccionados, ruta_salida_fig)

    print("\n Gráficos generados. Creando informe de texto (.md)...")
    


    # Preparación de datos para el informe 
    df_tir = df_largo[df_largo['DATAFIELD'] == 'TIR'].copy()
    df_duracion = df_largo[df_largo['DATAFIELD'] == 'DURACION']
    
    def get_basename(path):
        return os.path.basename(path) if path else 'N/A'
    
        

        
    fecha_hoy = pd.Timestamp.now().strftime('%Y-%m-%d')

    informe = f"""---
title: "Informe analítico de fondos de renta fija"
subtitle: "Análisis automatizado con Python"
author:
  - "Marcos Paulino Sicilia"
date: {fecha_hoy}
lang: es
keywords: [Renta Fija, Python, Análisis Cuantitativo, Regresión, Volatilidad]
toc: true
toc-depth: 3
number-sections: true
geometry: margin=1.5cm
mainfont: "Times New Roman"
fontsize: 11pt
---

# Introducción al informe

Este documento es un informe analítico generado automáticamente que detalla el comportamiento y las características de riesgo de los siguientes fondos de inversión: {', '.join(isins_seleccionados)}.

Descargo de responsabilidad

Este informe ha sido generado automáticamente. La información y los análisis presentados se basan únicamente en los datos históricos proporcionados y no deben ser considerados como asesoramiento financiero, de inversión o de cualquier otra índole. Las proyecciones y análisis técnicos son interpretaciones matemáticas de datos pasados y no garantizan resultados futuros.

"""

    
# Sección 1 
    try:
        informe += f"""
# Análisis de evolución histórica

![Evolución de TIR y Duración](fig/{get_basename(ruta_evolucion)})

Este análisis describe la trayectoria mensual de la TIR y la duración a lo largo del período de datos disponible.
"""

        df_tir_sec1 = df_largo[df_largo['DATAFIELD'] == 'TIR'].copy() if not df_largo.empty else pd.DataFrame()
        df_dur_sec1 = df_largo[df_largo['DATAFIELD'] == 'DURACION'].copy() if not df_largo.empty else pd.DataFrame()

        for isin in isins_seleccionados:
            informe += f"\n## {isin}\n"
            
            # Análisis de TIR
            if not df_tir_sec1.empty:
                tir_data = df_tir_sec1[df_tir_sec1['REF_ISIN'] == isin].sort_values('DATA_DATE')
                if len(tir_data) > 1:
                    primera_fecha_tir = tir_data['DATA_DATE'].iloc[0].strftime('%Y-%m')
                    ultima_fecha_tir = tir_data['DATA_DATE'].iloc[-1].strftime('%Y-%m')
                    primera_tir = tir_data['FUND_VALUE'].iloc[0]
                    ultima_tir = tir_data['FUND_VALUE'].iloc[-1]
                    max_tir, min_tir = tir_data['FUND_VALUE'].max(), tir_data['FUND_VALUE'].min()
                    tir_data['date_num'] = mdates.date2num(tir_data['DATA_DATE'])
                    slope_tir, _, _, _, _ = linregress(tir_data['date_num'], tir_data['FUND_VALUE'])
                    tendencia_tir = "alcista" if slope_tir > 1e-5 else "bajista" if slope_tir < -1e-5 else "estable" 

                    informe += f"""
- **TIR** (de {primera_fecha_tir} a {ultima_fecha_tir}): Mostró una tendencia general **{tendencia_tir}**. 
  Comenzó en {primera_tir:.4f} p.p. y finalizó en {ultima_tir:.4f} p.p. 
  El rango observado fue de {min_tir:.4f} a {max_tir:.4f} p.p.
"""
                else:
                    informe += "\n- **TIR**: Datos insuficientes para análisis."
            else:
                informe += "\n- **TIR**: No hay datos disponibles."

            # Análisis de duración
            if not df_dur_sec1.empty:
                dur_data = df_dur_sec1[df_dur_sec1['REF_ISIN'] == isin].sort_values('DATA_DATE')
                if len(dur_data) > 1:
                    primera_fecha_dur = dur_data['DATA_DATE'].iloc[0].strftime('%Y-%m')
                    ultima_fecha_dur = dur_data['DATA_DATE'].iloc[-1].strftime('%Y-%m')
                    primera_dur = dur_data['FUND_VALUE'].iloc[0]
                    ultima_dur = dur_data['FUND_VALUE'].iloc[-1]
                    max_dur, min_dur = dur_data['FUND_VALUE'].max(), dur_data['FUND_VALUE'].min()
                    dur_data['date_num'] = mdates.date2num(dur_data['DATA_DATE'])
                    slope_dur, _, _, _, _ = linregress(dur_data['date_num'], dur_data['FUND_VALUE'])
                    tendencia_dur = "alcista" if slope_dur > 1e-3 else "bajista" if slope_dur < -1e-3 else "estable" 

                    informe += f"""
- **Duración** (de {primera_fecha_dur} a {ultima_fecha_dur}): Mostró una tendencia general **{tendencia_dur}**.
  Comenzó en {primera_dur:.2f} años y finalizó en {ultima_dur:.2f} años.
  El rango observado fue de {min_dur:.2f} a {max_dur:.2f} años.
"""
                else:
                    informe += "\n- **Duración**: Datos insuficientes para análisis."
            else:
                    informe += "\n- **Duración**: No hay datos disponibles."

            informe += "\n" 

    except Exception as e:
        informe += f"\nERROR: No se pudo generar la sección de análisis de evolución histórica. Causa: {e}\n"
    informe += f"\n\n\\clearpage\n\n"
# Sección 2
    try:
        informe += f"""
# Análisis de dispersión riesgo/rendimiento

![Dispersión Riesgo/Rendimiento](fig/{get_basename(ruta_dispersion)})

Este gráfico posiciona cada activo en función de su rendimiento promedio (TIR promedio) y su riesgo de tipo de interés (duración promedio). La estrella roja marca la media del grupo.
"""
        
        promedios_tir = df_tir.groupby('REF_ISIN')['FUND_VALUE'].mean()
        promedios_duracion = df_duracion.groupby('REF_ISIN')['FUND_VALUE'].mean()
        media_grupo_tir = promedios_tir.mean()
        media_grupo_dur = promedios_duracion.mean()

        for isin in isins_seleccionados:
            informe += f"\n## {isin}\n"
            
            tir_isin, dur_isin = promedios_tir.get(isin, 0), promedios_duracion.get(isin, 0)
            pos_tir = "mayor" if tir_isin > media_grupo_tir else "menor"
            pos_dur = "mayor" if dur_isin > media_grupo_dur else "menor"
            perfil = ""
            if pos_tir == "mayor" and pos_dur == "mayor": perfil = "Rendimiento alto, riesgo alto"
            elif pos_tir == "mayor" and pos_dur == "menor": perfil = "Rendimiento alto, riesgo bajo (generalmente considerado un perfil atractivo)"
            elif pos_tir == "menor" and pos_dur == "mayor": perfil = "Rendimiento bajo, riesgo alto (generalmente considerado un perfil menos atractivo)"
            else: perfil = "Rendimiento bajo, riesgo bajo (perfil conservador)"
            
            informe += f"""
- **Posicionamiento**: Se sitúa en el cuadrante de **{pos_tir} TIR** y **{pos_dur} duración** en comparación con la media del grupo.
- **Perfil analítico**: Corresponde a un perfil de **"{perfil}"**. Su TIR promedio fue de {tir_isin:.4f} con una duración promedio de {dur_isin:.2f}.
"""
    except Exception as e:
        informe += f"\nERROR: No se pudo generar la sección de Análisis de Dispersión. Causa: {e}\n"
# Sección 3 
    try:
        informe += f"""
# Análisis de distribución y volatilidad

![Distribución de TIR y Duración](fig/{get_basename(ruta_boxplot)})

El gráfico de cajas muestra la mediana (línea central), los cuartiles (caja) y posibles valores atípicos (puntos) para TIR y duración, ofreciendo una visión de su estabilidad.
"""
        
        df_tir_sec3 = df_largo[df_largo['DATAFIELD'] == 'TIR'].copy() if not df_largo.empty else pd.DataFrame()
        df_dur_sec3 = df_largo[df_largo['DATAFIELD'] == 'DURACION'].copy() if not df_largo.empty else pd.DataFrame()

        for isin in isins_seleccionados:
            informe += f"\n## {isin}\n"
            
            # Análisis de TIR 
            if not df_tir_sec3.empty:
                tir_data = df_tir_sec3[df_tir_sec3['REF_ISIN'] == isin]['FUND_VALUE']
                if not tir_data.empty and tir_data.count() > 1:
                    q1_tir, median_tir, q3_tir = tir_data.quantile(0.25), tir_data.median(), tir_data.quantile(0.75)
                    iqr_tir = q3_tir - q1_tir
                    informe += f"""
- **Distribución de TIR**:
  - Mediana: {median_tir:.4f} puntos porcentuales.
  - Rango del 50% central (IQR): De {q1_tir:.4f} a {q3_tir:.4f} pp. (Amplitud IQR: {iqr_tir:.4f} pp).
  - *Volatilidad*: Un IQR más bajo sugiere mayor consistencia histórica en la TIR (los datos están más agrupados).
"""
                else:
                    informe += "\n- **Distribución de TIR**: Datos insuficientes."
            else:
                informe += "\n- **Distribución de TIR**: No hay datos disponibles."

            # Análisis de duración
            if not df_dur_sec3.empty:
                dur_data = df_dur_sec3[df_dur_sec3['REF_ISIN'] == isin]['FUND_VALUE']
                if not dur_data.empty and dur_data.count() > 1:
                    q1_dur, median_dur, q3_dur = dur_data.quantile(0.25), dur_data.median(), dur_data.quantile(0.75)
                    iqr_dur = q3_dur - q1_dur
                    informe += f"""
- **Distribución de duración**:
  - Mediana: {median_dur:.2f} años.
  - Rango del 50% central (IQR): De {q1_dur:.2f} a {q3_dur:.2f} años (Amplitud IQR: {iqr_dur:.2f} años).
  - *Consistencia del riesgo*: Un IQR bajo sugiere que el perfil de riesgo de tipo de interés ha sido más estable.
"""
                else:
                    informe += "\n- **Distribución de duración**: Datos insuficientes."
            else:
                    informe += "\n- **Distribución de duración**: No hay datos disponibles."

            informe += "\n" 
    except Exception as e:
        informe += f"\nERROR: No se pudo generar la sección de análisis de distribución. Causa: {e}\n"
        
# Sección 4 
    try:
        informe += f"""
# Análisis de correlación de retornos mensuales

![Heatmap de Correlación de Retornos](fig/{get_basename(ruta_heatmap_ret)})

La matriz de correlación mide cómo se mueven los **retornos mensuales** (calculados desde 'precio') de los activos entre sí. Un valor cercano a +1 indica que sus ganancias/pérdidas tienden a ocurrir simultáneamente; cerca de -1, tienden a moverse en direcciones opuestas; cerca de 0, sus movimientos no tienen una relación lineal clara. Este análisis es clave para evaluar la diversificación.
"""
        
        if 'precio' not in df_ancho.columns or df_ancho['precio'].isnull().all():
            informe += "\nNo hay datos de 'precio' suficientes para calcular la correlación de retornos.\n"
        elif len(isins_seleccionados) > 1:
            df_retornos = df_ancho[['DATA_DATE', 'REF_ISIN', 'precio']].copy()
            df_retornos.sort_values(by=['REF_ISIN', 'DATA_DATE'], inplace=True)
            df_retornos['retorno'] = df_retornos.groupby('REF_ISIN')['precio'].pct_change() * 100
            df_pivot_ret = df_retornos.pivot_table(index='DATA_DATE', columns='REF_ISIN', values='retorno')
            df_pivot_ret.dropna(axis=1, how='all', inplace=True)
            df_pivot_ret.dropna(axis=0, how='any', inplace=True) 

            if df_pivot_ret.shape[1] > 1:
                corr_matrix_ret = df_pivot_ret.corr()
                corr_unstacked_ret = corr_matrix_ret.unstack()
                corr_filtered_ret = corr_unstacked_ret[corr_unstacked_ret.index.get_level_values(0) != corr_unstacked_ret.index.get_level_values(1)]

                if not corr_filtered_ret.empty:
                    corr_sorted_ret = corr_filtered_ret.sort_values(kind="quicksort", ascending=False)
                    highest_corr_ret = corr_sorted_ret.head(1)
                    lowest_corr_ret = corr_sorted_ret.tail(1)

                    def clasificar_correlacion(r):
                        if pd.isna(r): return ""
                        abs_r = abs(r)
                        signo = "positiva" if r > 0 else "negativa" if r < 0 else "nula"
                        if abs_r >= 0.7: strength = "fuerte"
                        elif abs_r >= 0.4: strength = "moderada"
                        elif abs_r > 0.1: strength = "débil"
                        else: strength = "muy débil o nula"
                        if signo == "nula" or strength == "muy débil o nula": return f"(correlación {strength})"
                        return f"(correlación {signo} {strength})"

                    informe += f"""
- **Mayor correlación**: El par **{highest_corr_ret.index[0][0]} y {highest_corr_ret.index[0][1]}** (Coef: **{highest_corr_ret.values[0]:.2f}**) {clasificar_correlacion(highest_corr_ret.values[0])}. Sus retornos tienden a moverse muy sincronizados.
- **Menor correlación (o más negativa)**: El par **{lowest_corr_ret.index[0][0]} y {lowest_corr_ret.index[0][1]}** (Coef: **{lowest_corr_ret.values[0]:.2f}**) {clasificar_correlacion(lowest_corr_ret.values[0])}. Sus retornos tienen la menor sincronía (o la mayor tendencia a moverse en direcciones opuestas).
- **Implicación para diversificación**: Combinar activos cuyos **retornos** tengan baja correlación (idealmente < 0.4) ayuda a reducir la volatilidad de la cartera. Correlaciones altas (>0.7) indican poca diversificación entre esos activos.
"""
                else:
                    informe += "\nNo se pudieron calcular correlaciones válidas entre los retornos de los ISINs.\n"
            else:
                informe += "\nDatos insuficientes o no coincidentes en el tiempo para calcular correlación de retornos (tras eliminar NaNs).\n"
        elif len(isins_seleccionados) <= 1:
            informe += "\nSe necesita más de un ISIN para calcular la correlación.\n"

    except Exception as e:
        informe += f"\nERROR: No se pudo generar la sección de Análisis de correlación de retornos. Causa: {e}\n"

# Sección 5 
    try:
        informe += f"""
# Análisis de volatilidad móvil de la TIR

![Volatilidad Móvil de la TIR](fig/{get_basename(ruta_volatilidad_tir)})

Este gráfico muestra la evolución de la volatilidad (desviación estándar móvil de 12 meses) de la **TIR (yield)**. Mide la estabilidad del rendimiento ofrecido por el fondo, lo cual refleja la incertidumbre del mercado sobre tipos de interés y riesgo.
"""
        
        df_tir_sec5 = df_largo[df_largo['DATAFIELD'] == 'TIR'].copy() if not df_largo.empty else pd.DataFrame()

        if not df_tir_sec5.empty:
            ventana_vol_tir = 12 
            
            for isin in isins_seleccionados:
                informe += f"\n## {isin}\n"
                
                datos_isin_tir = df_tir_sec5[df_tir_sec5['REF_ISIN'] == isin].set_index('DATA_DATE').sort_index()['FUND_VALUE']
                datos_isin_tir = datos_isin_tir[~datos_isin_tir.index.duplicated(keep='last')]
                
                if len(datos_isin_tir) > ventana_vol_tir:
                    volatilidad_tir = datos_isin_tir.rolling(window=ventana_vol_tir).std()
                    avg_vol_tir = volatilidad_tir.mean()
                    max_vol_tir = volatilidad_tir.max()
                    
                    if not pd.isna(max_vol_tir) and max_vol_tir > 0:
                        max_vol_date_tir = volatilidad_tir.idxmax().strftime('%Y-%m')
                        informe += f"""
- **Volatilidad promedio (TIR)**: {avg_vol_tir:.4f} puntos porcentuales (p.p.).
- **Pico de volatilidad (TIR)**: Alcanzó su máxima inestabilidad de rendimiento alrededor de **{max_vol_date_tir}**, con un valor de {max_vol_tir:.4f} p.p.
- **Interpretación**: Picos de volatilidad en la TIR suelen coincidir con **mayor incertidumbre en el mercado** sobre tipos de interés o riesgo. Valles indican un entorno de rendimientos más estable.
"""
                    else:
                        informe += f"\n- **{isin}**: No se pudo determinar el pico de volatilidad de la TIR (datos insuficientes o nulos).\n"
                else:
                    informe += f"""
- **{isin}**:
  - **Volatilidad promedio (TIR)**: No se pudo calcular (datos insuficientes para una ventana de {ventana_vol_tir} meses).
"""
        else:
            informe += "\nNo hay datos de TIR disponibles para este análisis.\n"
            
    except Exception as e:
        informe += f"\nERROR: No se pudo generar la sección de Análisis de Volatilidad de TIR. Causa: {e}\n"

# Sección 6 
    try:
        informe += f"""
# Análisis de volatilidad móvil de los retornos

![Volatilidad Móvil de Retornos](fig/{get_basename(ruta_volatilidad_retornos)})

Este gráfico muestra la volatilidad (desviación estándar móvil de 12 meses) de los **retornos mensuales**. Mide el riesgo de precio que ha experimentado el inversor.
"""
        
        if 'precio' not in df_ancho.columns or df_ancho['precio'].isnull().all():
            informe += "\nNo hay datos de 'precio' suficientes para calcular la volatilidad de retornos.\n"
        else:
            df_ret_vol = df_ancho[['DATA_DATE', 'REF_ISIN', 'precio']].copy()
            df_ret_vol.sort_values(by=['REF_ISIN', 'DATA_DATE'], inplace=True)
            df_ret_vol['retorno'] = df_ret_vol.groupby('REF_ISIN')['precio'].pct_change() * 100
            
            ventana_vol_ret = 12 

            for isin in isins_seleccionados:
                informe += f"\n## {isin}\n"
                
                datos_isin_ret = df_ret_vol[df_ret_vol['REF_ISIN'] == isin].set_index('DATA_DATE')['retorno']
                datos_isin_ret = datos_isin_ret[~datos_isin_ret.index.duplicated(keep='last')].dropna()
                
                if len(datos_isin_ret) > ventana_vol_ret:
                    volatilidad_ret = datos_isin_ret.rolling(window=ventana_vol_ret).std()
                    avg_vol_ret = volatilidad_ret.mean()
                    max_vol_ret = volatilidad_ret.max()
                    
                    if not pd.isna(max_vol_ret) and max_vol_ret > 0:
                        max_vol_date_ret = volatilidad_ret.idxmax().strftime('%Y-%m')
                        informe += f"""
- **Volatilidad promedio (retornos)**: {avg_vol_ret:.2f}%.
- **Pico de volatilidad (retornos)**: Alcanzó su máxima inestabilidad de precio alrededor de **{max_vol_date_ret}**, con un valor de {max_vol_ret:.2f}%.
- **Interpretación**: Picos de volatilidad indican períodos de alto riesgo y retornos erráticos; valles indican estabilidad en los retornos.
"""
                    else:
                            informe += f"\n- **{isin}**: No se pudo determinar el pico de volatilidad de retornos (datos insuficientes o nulos).\n"
                else:
                    informe += f"""
- **{isin}**:
  - **Volatilidad promedio (retornos)**: No se pudo calcular (datos insuficientes para una ventana de {ventana_vol_ret} meses).
"""
        
    except Exception as e:
        informe += f"\nERROR: No se pudo generar la sección de Análisis de Volatilidad de Retornos. Causa: {e}\n"
        
# Sección 7 
    try:
        informe += f"""
# Análisis de distribución de cambios en la TIR

![Distribución de Cambios de TIR (KDE)](fig/{get_basename(ruta_cambios)})

Este análisis estudia la forma de la distribución de probabilidad (estimada mediante KDE) de los **cambios mensuales en la TIR (yield)**. A diferencia de un histograma, esta curva suavizada representa la densidad de probabilidad, ayudando a entender la probabilidad de movimientos bruscos en el rendimiento.
"""
    
        df_tir_sec7 = df_largo[df_largo['DATAFIELD'] == 'TIR'].copy() if not df_largo.empty else pd.DataFrame()

        if not df_tir_sec7.empty:
            for isin in isins_seleccionados:
                informe += f"\n## {isin}\n"
                
                datos_isin = df_tir_sec7[df_tir_sec7['REF_ISIN'] == isin].set_index('DATA_DATE').sort_index()['FUND_VALUE']
                datos_isin = datos_isin[~datos_isin.index.duplicated(keep='last')] 

                cambios_mensuales = datos_isin.diff().dropna()
                
                if len(cambios_mensuales) > 3: 
                
                    std_cambio = cambios_mensuales.std()
                    kurt = kurtosis(cambios_mensuales) 
                    sk = skew(cambios_mensuales)
                    
                    if abs(sk) > 0.5:
                        tipo_asimetria = "Fuertemente asimétrica"
                        intensidad_asimetria = "ALTA"
                    elif abs(sk) > 0.1:
                        tipo_asimetria = "Moderadamente asimétrica" 
                        intensidad_asimetria = "MEDIA"
                    else:
                        tipo_asimetria = "Básicamente simétrica"
                        intensidad_asimetria = "BAJA"
                    
                    direccion_asimetria = "positiva (cambios al alza más extremos)" if sk > 0 else "negativa (cambios a la baja más extremos)" if sk < 0 else "simétrica"
                    
                    if kurt > 1:
                        tipo_curtosis = "Leptocúrtica (colas pesadas)"
                        riesgo_cola = "ALTO"
                    elif kurt > -0.5:
                        tipo_curtosis = "Mesocúrtica (similar a normal)"
                        riesgo_cola = "MODERADO"
                    else:
                        tipo_curtosis = "Platicúrtica (colas ligeras)"
                        riesgo_cola = "BAJO"
                    
                    cambio_maximo = cambios_mensuales.max()
                    cambio_minimo = cambios_mensuales.min()
                    
                    informe += f"""
- **ANÁLISIS DE DISTRIBUCIÓN** (Basado en {len(cambios_mensuales)} meses):
  - **Asimetría (Skewness)**: {sk:.3f} - {tipo_asimetria} ({direccion_asimetria})
  - **Curtosis (Exceso)**: {kurt:.3f} - {tipo_curtosis}
  - **Riesgo de cola**: {riesgo_cola} - probabilidad de movimientos extremos en TIR
"""
                    
                    recomendaciones = []
                    
                    if riesgo_cola == "ALTO":
                        recomendaciones.append("Alto riesgo de eventos extremos - considerar coberturas")
                    if intensidad_asimetria == "ALTA":
                        if sk > 0:
                            recomendaciones.append("Sesgo positivo: mayor probabilidad de saltos bruscos al alza en TIR")
                        else:
                            recomendaciones.append("Sesgo negativo: mayor probabilidad de caídas bruscas en TIR")
                    
                    if std_cambio > 0 and (abs(cambio_maximo) > 3 * std_cambio or abs(cambio_minimo) > 3 * std_cambio):
                        recomendaciones.append("Presencia de movimientos extremos (>3σ) detectados")
                    
                    if recomendaciones:
                        informe += "- **RECOMENDACIONES**:\n" + "\n".join(f"  - {rec}" for rec in recomendaciones)

                        informe += "\n"

                        
                else: 
                    informe += f"\n- **{isin}**: Datos insuficientes para análisis de distribución de cambios de TIR.\n"
        else:
            informe += "\nNo hay datos de TIR disponibles para este análisis.\n"
            
    except Exception as e:
        informe += f"\nERROR: No se pudo generar la sección de análisis de distribución de cambios en la TIR. Causa: {e}\n"



# Sección 8
    try:
        informe += f"""
# Análisis de comparativa frente a índice
Este análisis compara el **retorno mensual del fondo** contra el retorno mensual de su índice de referencia asignado. Ambos retornos están expresados en **formato porcentaje (ej: 1.5%)**.
"""
        
        if df_resultados_comparativa.empty:
            informe += "\nNo se generaron datos de comparativa. Esto puede ocurrir si los fondos no fueron clasificados o si el fichero de índices no se encontró.\n"
        else:
            for _, row in df_resultados_comparativa.iterrows():
                isin = row['ISIN']
                indice_ref = row['Indice_Referencia']
                corr = row['Correlacion']
                te = row['Tracking_Error'] 

                corr_desc = "N/A"
                if pd.isna(corr):
                    corr_desc = "No se pudo calcular."
                elif corr > 0.85:
                    corr_desc = f"**muy fuerte y positiva ({corr:.2f})**: El fondo sigue muy de cerca a su índice."
                elif corr > 0.6:
                    corr_desc = f"**fuerte y positiva ({corr:.2f})**: El fondo tiende a moverse en la misma dirección que su índice."
                else:
                    corr_desc = f"**baja/moderada ({corr:.2f})**: El fondo no sigue de cerca al índice."

                te_desc = "N/A"
                if not pd.isna(te):
                    te_desc = f"**{te:.2f}% mensual**. un T.E. bajo (< 0.5%) sugiere gestión pasiva. Un T.E. alto (> 1.5%) indica gestión activa."


                informe += f"""

## {isin} vs. {indice_ref}

![Comparativa Fondo vs Índice para {isin}](fig/comparativa_{isin}.png)

- **Correlación de retornos**: {corr_desc}
- **Tracking error**: {te_desc}
"""

    except Exception as e:
        informe += f"\nERROR: No se pudo generar la sección de Comparativa con Índices. Causa: {e}\n"
    
    informe += f"\n\n\\clearpage\n\n"


# Sección 9
    try:
        informe += f"""
# Análisis avanzado de volatilidad (Ft adapted volatility)

Este análisis aplica una metodología avanzada (**Ft adapted volatility**) basada en la teoría de la **inatención racional**. A diferencia de la volatilidad móvil tradicional, este modelo utiliza un proceso **autoarima** para filtrar los retornos (eliminando autocorrelación) y estima la volatilidad basándose en los residuos o "shocks" puros.

El modelo estima un parámetro **Phi (φ)** para cada fondo, que representa la **"probabilidad de llegada de noticias"**:
 un **φ alto** indica que el fondo es muy sensible a lo reciente ("memoria corta").
 Un **φ bajo** indica que el riesgo es más estructural y estable ("memoria larga").

![Evolución de la Volatilidad Adaptada](fig/{get_basename(ruta_ft_vol)})

## Interpretación del parámetro de atención (φ):
"""
        if dict_phis:
            for isin, phi in dict_phis.items():
                # Obtenemos el modelo ARIMA calculado para este fondo
                modelo_usado = dict_arimas.get(isin, "N/A")
                
                # Interpretación simple
                if phi > 0.10:
                    interpretacion = "Alta reactividad. Muy sensible a noticias recientes."
                elif phi > 0.05:
                    interpretacion = "Reactividad moderada."
                else:
                    interpretacion = "Baja reactividad. Riesgo estructural estable."
                
                # Formato: ISIN: Phi [Modelo]. Interpretación.
                informe += f"- **{isin}**: φ = {phi:.4f} [**{modelo_usado}**]. {interpretacion}\n"
        else:
            informe += "\nNo se pudo calcular el parámetro phi ni el modelo ARIMA (datos insuficientes).\n"
            
        informe += """
**Conclusión del modelo:**

* **Selección del modelo:** El algoritmo ha seleccionado automáticamente el mejor modelo ARIMA(p,d,q) para cada fondo minimizando el criterio AIC, asegurando que la volatilidad se calcula sobre los *shocks* reales.
* **Dinámica del riesgo:** Esta métrica permite detectar **cambios de régimen** (saltos de volatilidad) de forma casi instantánea, ajustándose dinámicamente a las condiciones de mercado en lugar de suavizarlas.
"""

    except Exception as e:
        informe += f"\nERROR: No se pudo generar la sección de Ft adapted volatility. Causa: {e}\n"

    informe += f"\n\n\\clearpage\n\n"


# Sección 10
    try:
        informe += f"""
# Análisis de regresiones macroeconómicas
En esta sección se analizan las regresiones múltiples realizadas entre el **retorno mensual del fondo** (variable dependiente) y tres variables explicativas: el **retorno de su índice de referencia**, el **nivel del tipo de interés** y el **nivel de inflación** correspondientes. Los resultados numéricos (coeficientes, p-values, R²) se encuentran en el fichero `resultados_regresion_macro.csv`. El mapeo de variables usado se encuentra en `mapeo_variables_macro.csv`.
"""

        ruta_macro_resultados = os.path.join(ruta_salida, "resultados_regresion_macro.csv")
        ruta_macro_mapeo = os.path.join(ruta_salida, "mapeo_variables_macro.csv")
        df_resultados_macro = pd.DataFrame() 
        
        if os.path.exists(ruta_macro_resultados) and os.path.exists(ruta_macro_mapeo):
            try:
                df_resultados_leido = pd.read_csv(ruta_macro_resultados, sep=";", encoding="ansi", decimal=".")
                df_mapeo_leido = pd.read_csv(ruta_macro_mapeo, sep=";", encoding="ansi", decimal=".")
                
                df_resultados_macro = pd.merge(
                    df_resultados_leido, 
                    df_mapeo_leido[['ISIN', 'Variable_Indice', 'Variable_TipoInteres', 'Variable_Inflacion']], 
                    on='ISIN', 
                    how='left'
                )
                df_resultados_macro.rename(
                    columns={'Variable_Indice': 'Indice_Referencia'}, 
                    inplace=True
                )

            except Exception as e_read_csv:
                print(f" Error leyendo los CSVs de regresión: {e_read_csv}")
                df_resultados_macro = pd.DataFrame() 
        
        else:
            print(" No se encontraron archivos de resultados de regresión (CSV).")


        if not df_resultados_macro.empty:
            
            informe += "\n## Análisis de Sensibilidad Individual por Fondo (ISIN)\n"
            
            for _, fila in df_resultados_macro.iterrows():
                if fila['ISIN'] in isins_seleccionados:
                    
                    sig_alpha = " (Estadísticamente significativo, p < 0.05)" if fila['p_valor_Alpha'] < 0.05 else " (No significativo)"
                    sig_beta_idx = " (Estadísticamente significativo, p < 0.05)" if fila['p_valor_Beta_Indice'] < 0.05 else " (No significativo)"
                    sig_beta_tipos = " (Estadísticamente significativo, p < 0.05)" if fila['p_valor_Beta_TipoInteres'] < 0.05 else " (No significativo)"
                    sig_beta_infl = " (Estadísticamente significativo, p < 0.05)" if fila['p_valor_Beta_Inflacion'] < 0.05 else " (No significativo)"
                    
                    nombre_indice = fila.get('Indice_Referencia', 'Índice')
                    nombre_tipo_interes = fila.get('Variable_TipoInteres', 'Tipo de Interés')
                    nombre_inflacion = fila.get('Variable_Inflacion', 'Inflación')
                    
                    informe += f"""
### {fila['ISIN']} 
- **R-cuadrado ({fila['R2']:.2f})**:
  - *Análisis*: Un **{fila['R2']*100:.0f}%** de la variación de los retornos de este fondo es explicado por el modelo (índice, tipos, inflación). Un R² alto (>0.7) implica alta dependencia de estos factores.
"""
                    
                    if 'Alpha' in fila and pd.notna(fila['Alpha']):
                        informe += f"""- **Alfa (gestión activa) ({fila['Alpha']:.2f}%)** - {sig_alpha}:
  - *Interpretación*: Es el retorno mensual promedio que **no** es explicado por los factores de riesgo (índice, tipos, inflación). Mide la habilidad del gestor.
"""
                        if fila['Alpha'] > 0.05 and fila['p_valor_Alpha'] < 0.05:
                            informe += f"  - *Análisis del coeficiente*: **Alfa positivo y significativo**. El fondo ha generado un valor extra del {fila['Alpha']:.2f}% mensual por encima de lo esperado.\n"
                        elif fila['Alpha'] < -0.05 and fila['p_valor_Alpha'] < 0.05:
                            informe += f"  - *Análisis del coeficiente*: **Alfa negativo y significativo**. El fondo ha rendido un {abs(fila['Alpha']):.2f}% mensual por debajo de lo esperado.\n"
                        else:
                            informe += "  - *Análisis del coeficiente*: **Alfa no significativo**. El rendimiento del fondo está en línea con lo explicado por sus factores de riesgo (Alfa = 0).\n"
                    
                    informe += f"""- **Beta vs. {nombre_indice} ({fila['Beta_Indice']:.2f}%)** - {sig_beta_idx}:
  - *Interpretación*: Por cada 1% de retorno del índice, este fondo obtiene un **{fila['Beta_Indice']:.2f}%** de retorno.
"""
                    if 0.95 < fila['Beta_Indice'] < 1.05:
                        informe += "  - *Análisis del coeficiente*: Valor **cercano a 1.0**, sugiere un alto seguimiento del benchmark.\n"
                    elif fila['Beta_Indice'] > 1.05:
                        informe += f"  - *Análisis del coeficiente*: Valor **superior a 1.0**, indica un perfil más volátil que el índice.\n"
                    else: 
                        informe += f"  - *Análisis del coeficiente*: Valor **inferior a 1.0**, sugiere un perfil más defensivo.\n"

                    informe += f"""- **Beta vs. {nombre_tipo_interes} ({fila['Beta_TipoInteres']:.2f}%)** - {sig_beta_tipos}:
  - *Interpretación*: Por cada subida de 1 punto (100 pbs) en el tipo de interés, el retorno mensual del fondo cambia un **{fila['Beta_TipoInteres']:.2f}%**.
"""
                    if fila['Beta_TipoInteres'] < -0.1:
                        informe += "  - *Análisis del coeficiente*: **Negativo**, confirma la relación inversa (sensibilidad a tipos).\n"
                    elif fila['Beta_TipoInteres'] > 0.1:
                        informe += "  - *Análisis del coeficiente*: **Positivo**, atípico (posible duración corta o flotante).\n"
                    else:
                        informe += "  - *Análisis del coeficiente*: **Cercano a cero**, sensibilidad casi nula a los tipos.\n"
                    
                    informe += f"""- **Beta vs. {nombre_inflacion} ({fila['Beta_Inflacion']:.2f}%)** - {sig_beta_infl}:
  - *Interpretación*: Por cada subida de 1 punto en la inflación, el retorno mensual cambia un **{fila['Beta_Inflacion']:.2f}%**.
"""
                    if fila['Beta_Inflacion'] > 0.1:
                        informe += "  - *Análisis del coeficiente*: **Positivo**, sugiere cierta **cobertura** contra la inflación.\n"
                    elif fila['Beta_Inflacion'] < -0.1:
                        informe += "  - *Análisis del coeficiente*: **Negativo**, la inflación perjudica el retorno.\n"
                    else:
                        informe += "  - *Análisis del coeficiente*: **Cercano a cero**, la inflación no es un factor relevante.\n"

            informe += """
## Conclusión general del modelo macro
El análisis de regresiones permite cuantificar las fuentes de retornos de los fondos:

1.  **Beta vs. Índice**: Muestra qué parte del retorno se debe al seguimiento del mercado (benchmark).
2.  **Beta vs. Tipo de Interés**: Es el factor macroeconómico clave. Cuantifica la sensibilidad del retorno a cambios en la política monetaria (duración).
3.  **Beta vs. Inflación**: Mide la capacidad del fondo para proteger el retorno en entornos inflacionarios.
4.  **Alfa**: Mide el retorno extra (o faltante) que no se debe a estos factores. Es la métrica clave para evaluar la habilidad de gestión activa.
5.  **P-value (significatividad)**: Nos dice si podemos confiar en estos coeficientes. Un p-value < 0.05 sugiere que el factor es estadísticamente relevante para explicar el retorno.

El **R²** (R-cuadrado) indica el grado de éxito de este modelo. Un R² alto (ej. > 70%) significa que estos tres factores (más el Alfa) explican la gran mayoría de los movimientos del fondo.
"""
        else:
            informe += "\nNo se encontraron resultados de regresiones macroeconómicas para mostrar (archivos CSV no encontrados).\n"

    except Exception as e:
        informe += f"\nERROR: No se pudo generar la sección de análisis de regresiones macroeconómicas. Causa: {e}\n"
    
    
    ruta_informe_md = os.path.join(ruta_salida, "informe_analitico.md")
    try:
        with open(ruta_informe_md, 'w', encoding='utf-8') as f:
            f.write(informe)
        print(f"\n+ Informe de análisis (.md) guardado como: {ruta_informe_md}")
    except Exception as e:
        print(f"\nError al guardar el informe .md: {e}")
        return None 

    return ruta_informe_md

# 6. BLOQUE PRINCIPAL DE EJECUCIÓN 
if __name__ == '__main__':
    # Pon tu ruta de histórico aquí
    RUTA_HISTORICO = r"C:\Users\Usuario\Desktop\Tfg Economía"
    RUTA_SALIDA = r"C:\Users\Usuario\Desktop\Tfg Economía\salida"
    MASCARA_QF = "*_ficheroQF.csv"

    # Ruta al fichero de índices 
    RENTAB_INDICES_FILE_PATH = r"C:\Users\Usuario\Desktop\Tfg Economía\Rentab Indices RF.xlsx"
    

    # Ruta para las figuras (usada por Pandoc)
    RUTA_FIG = os.path.join(RUTA_SALIDA, "fig")

    if not os.path.exists(RUTA_SALIDA):
        os.makedirs(RUTA_SALIDA)
        print(f"Directorio de salida principal creado en: {RUTA_SALIDA}")
    if not os.path.exists(RUTA_FIG):
        os.makedirs(RUTA_FIG)
        print(f"Directorio de figuras creado en: {RUTA_FIG}")


    ruta_completa_pre = os.path.join(RUTA_HISTORICO, MASCARA_QF)
    ficheros_pre = sorted(glob.glob(ruta_completa_pre))
    
    if not ficheros_pre:
        print(f"No se encontraron ficheros en '{RUTA_HISTORICO}' con la máscara '{MASCARA_QF}'")
    else:
        lista_isin_disponibles = []
        for f in ficheros_pre:
            try:
                df_isin_temp = pd.read_csv(f, sep=';', encoding='latin1', on_bad_lines='skip', usecols=['ISIN_SHARE_CLASS'])
                df_isin_temp.rename(columns={'ISIN_SHARE_CLASS': 'REF_ISIN'}, inplace=True)
                lista_isin_disponibles.append(df_isin_temp)
            except (ValueError, KeyError):
                pass
            except Exception as e:
                print(f"Advertencia: Error leyendo {f}. Error: {e}")

        if not lista_isin_disponibles:
            df_disponible = pd.DataFrame(columns=['REF_ISIN'])
        else:
            df_disponible = pd.concat(lista_isin_disponibles).drop_duplicates().reset_index(drop=True)

        isins_seleccionados = seleccionar_origen_isins(df_disponible)

        if isins_seleccionados:
            df_ancho_final, df_largo_final, df_td_raw, df_precio_raw = cargar_datos_en_chunks(
                RUTA_HISTORICO, 
                MASCARA_QF, 
                isins_seleccionados
                )        
            
            # Guardar históricos (se guardan en RUTA_SALIDA, no en 'fig')
            guardar_historicos_separados(df_ancho_final, df_precio_raw, isins_seleccionados, RUTA_SALIDA)


            # Clasificación 
            print("\n Clasificando fondos según región, divisa y media de duración")
            
            print("Calculando media de duración y datos estáticos por ISIN...")
            df_clasificacion = df_ancho_final.groupby('REF_ISIN').agg(
                category_region=('category_region', 'first'), 
                nav_crncy=('nav_crncy', 'first'),
                Media_DURACION=('DURACION', 'mean') 
            ).reset_index()


            def asignar_indice(row):
                region = str(row.get("category_region", "")).upper().strip()
                currency = str(row.get("nav_crncy", "")).upper().strip()
                
                if not region or not currency:
                    return "Sin índice asignado"

                # EUROPA 
                if region == "EUROPA" and currency == "EUR":
                    return "Bloomberg Euro Agg Bond TR EUR"
                # EEUU
                elif region == "EEUU" and currency == "USD":
                    return "Bloomberg US Agg Bond TR USD"
                # GLOBAL
                elif region == "GLOBAL":
                    if currency == "USD":
                        return "Bloomberg Global Aggregate TR Hdg USD"
                    elif currency == "EUR":
                        return "Bloomberg Global Aggregate TR Hdg EUR"
                return "Sin índice asignado"

            df_clasificacion["Indice_Referencia"] = df_clasificacion.apply(asignar_indice, axis=1)

            df_clasificacion_para_guardar = df_clasificacion[[
                'REF_ISIN', 'Media_DURACION', 'category_region', 'nav_crncy', 'Indice_Referencia'
            ]]
            

            
            df_ancho_final = pd.merge(
                df_ancho_final, 
                df_clasificacion[['REF_ISIN', 'Indice_Referencia']], 
                on='REF_ISIN', 
                how='left'
            )
            
            resumen_indices = df_clasificacion
            
            #  Comparación con índices
            print("\n Leyendo índices y generando comparativas (RETORNO % vs RETORNO %)")

            df_resultados = pd.DataFrame() 

            try:
                
                df_indices = pd.read_excel(
                    RENTAB_INDICES_FILE_PATH,
                    decimal='.',
                )
                df_indices.rename(columns={df_indices.columns[0]: "Fecha"}, inplace=True)
                df_indices["Fecha"] = pd.to_datetime(df_indices["Fecha"], errors='coerce')


                dic_indices = {
                    "Bloomberg Euro Agg Bond TR EUR": "Bloomberg Euro Agg Bond TR EUR",
                    "Bloomberg US Agg Bond TR USD": "Bloomberg US Agg Bond TR USD",
                    "Bloomberg Global Aggregate TR Hdg USD": "Bloomberg Global Aggregate TR Hdg USD",
                    "Bloomberg Global Aggregate TR Hdg EUR": "Bloomberg Global Aggregate TR Hdg EUR",
                }

                resultados = []

                for _, fila in resumen_indices.iterrows():
                    if fila["REF_ISIN"] not in isins_seleccionados:
                        continue
                        
                    isin = fila["REF_ISIN"]
                    indice_ref = fila["Indice_Referencia"]

                    if indice_ref not in dic_indices:
                        print(f" ISIN {isin} sin índice asignado ('{indice_ref}') o índice no en diccionario. Saltando.")
                        continue

                    df_fondo = df_ancho_final[df_ancho_final["REF_ISIN"] == isin].copy()
                    df_fondo['Monthly_Return_Fondo'] = df_fondo['precio'].pct_change() * 100
                    df_fondo.rename(columns={'DATA_DATE': 'Fecha'}, inplace=True) 
                    df_fondo["Fecha"] = pd.to_datetime(df_fondo["Fecha"], errors='coerce')
                    df_fondo.sort_values("Fecha", inplace=True)

                    col_indice = dic_indices[indice_ref]
                    if col_indice not in df_indices.columns:
                        print(f" Columna '{col_indice}' no encontrada en el Excel de índices '{RENTAB_INDICES_FILE_PATH}'.")
                        continue

                    df_indice = df_indices[["Fecha", col_indice]].copy().dropna()
                    df_fondo["Fecha"] = df_fondo["Fecha"] + pd.offsets.MonthEnd(0)
                    df_indice["Fecha"] = df_indice["Fecha"] + pd.offsets.MonthEnd(0)
                    df_merge = pd.merge(df_fondo, df_indice, on="Fecha", how="inner", suffixes=("_fondo", "_indice"))
                    df_merge = df_merge.dropna(subset=['Monthly_Return_Fondo', col_indice])

                    if df_merge.empty:
                        print(f" No hay fechas coincidentes (o datos de retorno) para {isin} y su índice {indice_ref}.")
                        continue

                    corr = df_merge["Monthly_Return_Fondo"].corr(df_merge[col_indice])
                    tracking_error = (df_merge["Monthly_Return_Fondo"] - df_merge[col_indice]).std() 
                    
                    resultados.append({
                        "ISIN": isin,
                        "Indice_Referencia": indice_ref,
                        "Correlacion": round(corr, 3) if not pd.isna(corr) else None,
                        "Tracking_Error": round(tracking_error, 3) if not pd.isna(tracking_error) else None,
                    })

                    print(f" Generando gráfico PNG de RETORNOS para {isin}...")
                    plt.figure(figsize=(10, 5))
                    plt.plot(df_merge["Fecha"], df_merge["Monthly_Return_Fondo"], label=f"Fondo ({isin}) - Retorno Mensual", alpha=0.8, marker='o', markersize=4)
                    plt.plot(df_merge["Fecha"], df_merge[col_indice], label=f"Índice ({indice_ref}) - Retorno Mensual", linestyle='--', marker='x', markersize=4)
                    plt.title(f"Comparativa retornos mensuales (%): Fondo vs Índice - {isin}")
                    plt.xlabel("Fecha")
                    plt.ylabel("Retorno Mensual (%)")
                    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1f}%'.format(y)))
                    plt.legend()
                    plt.grid(True)
                    
                    # Guardar gráfico en la carpeta 'fig'
                    grafico_path = os.path.join(RUTA_FIG, f"comparativa_{isin}.png")
                    plt.savefig(grafico_path, dpi=150, bbox_inches="tight")
                    plt.close()

                df_resultados = pd.DataFrame(resultados)
                ruta_resultados = os.path.join(RUTA_SALIDA, "comparativa_fondo_indice.csv")
                df_resultados.to_csv(ruta_resultados, sep=';', decimal='.', index=False, encoding='ansi', float_format='%.3f')

                print(f" Comparativas de RETORNOS (%) generadas y guardadas en:\n{ruta_resultados}")
            
            except FileNotFoundError:
                print(f"\n ERROR CRÍTICO: No se encontró el fichero de índices en la ruta especificada:")
                print(f"  {RENTAB_INDICES_FILE_PATH}")
            except Exception as e:
                print(f"\n ERROR inesperado durante la comparación con índices: {e}")

            #  REGRESIONES CON TIPOS DE INTERÉS E INFLACIÓN

            print("\n Añadiendo tipos de interés e inflación y ejecutando regresiones macroeconómicas")

            try:
                ruta_tipos = os.path.join(RUTA_HISTORICO, "tipointeres.xlsx")
                ruta_inflacion = os.path.join(RUTA_HISTORICO, "inflacion.xlsx")

                df_tipos = pd.read_excel(ruta_tipos, decimal='.')
                df_inflacion = pd.read_excel(ruta_inflacion, decimal='.')

                df_tipos.rename(columns={df_tipos.columns[0]: "Fecha"}, inplace=True)
                df_tipos["Fecha"] = pd.to_datetime(df_tipos["Fecha"], errors="coerce") + pd.offsets.MonthEnd(0)

                df_inflacion.rename(columns={df_inflacion.columns[0]: "Fecha"}, inplace=True)
                df_inflacion["Fecha"] = pd.to_datetime(df_inflacion["Fecha"], errors="coerce") + pd.offsets.MonthEnd(0)

                dic_macro = {
                    "Bloomberg Euro Agg Bond TR EUR": {
                        "tipo_corto": "Tipointeres_Europa_corto", 
                        "tipo_largo": "Tipointeres_Europa_largo", 
                        "inflacion": "Inflacion_Europa"
                    },
                    "Bloomberg US Agg Bond TR USD": {
                        "tipo_corto": "Tipointeres_EEUU_corto", 
                        "tipo_largo": "Tipointeres_EEUU_largo", 
                        "inflacion": "Inflacion_EEUU"
                    },
                    "Bloomberg Global Aggregate TR Hdg USD": {
                        "tipo_corto": "Tipointeres_EEUU_corto", 
                        "tipo_largo": "Tipointeres_EEUU_largo", 
                        "inflacion": "Inflacion_EEUU"
                    },
                    "Bloomberg Global Aggregate TR Hdg EUR": {
                        "tipo_corto": "Tipo_interes_Global_corto_eur", 
                        "tipo_largo": "Tipo_interes_Global_largo_eur", 
                        "inflacion": "Inflacion_Mundial"
                    },
                }

                resultados_macro = [] 
                mapeo_variables = []  

                for _, fila in resumen_indices.iterrows():
                    if fila["REF_ISIN"] not in isins_seleccionados:
                        continue
                        
                    try:
                        isin = fila["REF_ISIN"]
                        indice_ref = fila["Indice_Referencia"]
                        duracion_media_fondo = fila.get("Media_DURACION", None)

                        if indice_ref not in dic_indices or indice_ref not in dic_macro:
                            continue
                        
                        # Necesitamos la duración media para elegir el tipo de interés
                        if duracion_media_fondo is None or pd.isna(duracion_media_fondo):
                            print(f"Advertencia: No se pudo obtener duración media para {isin}. Saltando regresión macro.")
                            continue

                        df_fondo = df_ancho_final[df_ancho_final["REF_ISIN"] == isin].copy()
                        df_fondo.rename(columns={'DATA_DATE': 'Fecha'}, inplace=True)
                        df_fondo["Fecha"] = pd.to_datetime(df_fondo["Fecha"], errors="coerce")
                        df_fondo.sort_values("Fecha", inplace=True)
                        
                        df_fondo['Monthly_Return_Fondo'] = df_fondo['precio'].pct_change() * 100

                        col_indice = dic_indices[indice_ref]
                        if col_indice not in df_indices.columns:
                            continue
                        df_indices["Fecha"] = pd.to_datetime(df_indices["Fecha"], errors="coerce") + pd.offsets.MonthEnd(0)
                        
                        UMBRAL_DURACION = 2 
                        variables_macro = dic_macro[indice_ref]
                        
                        if duracion_media_fondo <= UMBRAL_DURACION:
                            col_tipo = variables_macro["tipo_corto"]
                        else:
                            col_tipo = variables_macro["tipo_largo"]
                            
                        col_infl = variables_macro["inflacion"]


                        mapeo_variables.append({
                            "ISIN": isin,
                            "Region": fila["category_region"],
                            "Divisa": fila["nav_crncy"],
                            "Variable_Indice": col_indice,
                            "Variable_TipoInteres": col_tipo, 
                            "Variable_Inflacion": col_infl
                        })

                        if col_tipo not in df_tipos.columns or col_infl not in df_inflacion.columns:
                            print(f"Advertencia: Columnas macro {col_tipo} o {col_infl} no encontradas. Saltando regresión para {isin}.")
                            continue
                        
                        df_macro = df_tipos[["Fecha", col_tipo]].merge(df_inflacion[["Fecha", col_infl]], on="Fecha", how="inner")

                        df_merge = (
                            df_fondo.merge(df_indices, on="Fecha", how="inner", suffixes=("_fondo", "_indice"))
                            .merge(df_macro, on="Fecha", how="left")
                        )

                        if df_merge.empty or 'Monthly_Return_Fondo' not in df_merge.columns:
                            continue

                        X = df_merge[[col_indice, col_tipo, col_infl]].copy()
                        X = sm.add_constant(X)
                        y = df_merge["Monthly_Return_Fondo"]
                        
                        df_modelo_data = pd.concat([y, X], axis=1).dropna()
                        if df_modelo_data.shape[0] < (X.shape[1] + 1): 
                            print(f"  Datos insuficientes para regresión macro de {isin} tras eliminar NaNs.")
                            continue
                        
                        y_clean = df_modelo_data[y.name]
                        X_clean = df_modelo_data[X.columns]
                        
                        modelo = sm.OLS(y_clean, X_clean).fit()
                        
                        resultados_macro.append({
                            "ISIN": isin,
                            "Alpha": round(modelo.params.get('const', 0), 4),
                            "p_valor_Alpha": round(modelo.pvalues.get('const', 0), 4),
                            "Beta_Indice": round(modelo.params.get(col_indice, 0), 4),
                            "p_valor_Beta_Indice": round(modelo.pvalues.get(col_indice, 0), 4),
                            "Beta_TipoInteres": round(modelo.params.get(col_tipo, 0), 4),
                            "p_valor_Beta_TipoInteres": round(modelo.pvalues.get(col_tipo, 0), 4),
                            "Beta_Inflacion": round(modelo.params.get(col_infl, 0), 4),
                            "p_valor_Beta_Inflacion": round(modelo.pvalues.get(col_infl, 0), 4),
                            "R2": round(modelo.rsquared, 4),
                            "Num_Obs": int(modelo.nobs)
                        })
                        
                    except Exception as e_reg:
                        print(f"ERROR durante la regresión macro para {isin}: {e_reg}")
                        continue

                # Guardar dataframes en dos CSVs separados
                df_resultados_macro = pd.DataFrame(resultados_macro)
                df_mapeo_variables = pd.DataFrame(mapeo_variables)
                
                ruta_macro_csv = os.path.join(RUTA_SALIDA, "resultados_regresion_macro.csv")
                ruta_mapeo_csv = os.path.join(RUTA_SALIDA, "mapeo_variables_macro.csv")

                try:
                    df_resultados_macro.to_csv(ruta_macro_csv, sep=";", decimal='.', index=False, encoding="ansi")
                    df_mapeo_variables.to_csv(ruta_mapeo_csv, sep=";", decimal='.', index=False, encoding="ansi")
                    print(f" Regresiones generadas en (dos archivos CSV):\n1: {ruta_macro_csv}\n2: {ruta_mapeo_csv}")
                
                except Exception as e_csv:
                    print(f" ERROR al guardar archivos CSV de regresión: {e_csv}")

            except FileNotFoundError as e_file:
                print(f"\n ERROR CRÍTICO: No se encontró un fichero macro (tipos o inflación). Revisa las rutas.")
                print(f"  Detalle: {e_file}")
            except Exception as e_main_macro:
                print(f"\n ERROR inesperado en el bloque de regresión macro: {e_main_macro}")


            #  INFORME ANALÍTICO

            if df_ancho_final.empty or df_largo_final.empty:
                print("\nAdvertencia: No se pudieron generar los análisis y gráficos porque no hay datos unificados.")
            else:
                

                ruta_informe_md = generar_informe_analitico(df_largo_final, df_ancho_final, isins_seleccionados, df_resultados, RUTA_SALIDA, RUTA_FIG)


                if ruta_informe_md:
                    llamar_pandoc(ruta_informe_md, RUTA_SALIDA)

            print("\n Proceso completado")
        else:
            print("No se seleccionaron ISINs. Fin del programa.")