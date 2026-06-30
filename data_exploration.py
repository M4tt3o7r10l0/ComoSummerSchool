#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import re
import unicodedata
from xml.etree import ElementTree as ET
from zipfile import ZipFile

DATA = 'data/'

# %%
data = gpd.read_file(
    DATA+'limiti_comunali_2020.geojson'
)
data = data.drop(columns='anno')
data.head()

# %%
data.plot()

#%%
istat = pd.read_csv(DATA + 'istat_codes.csv', sep=';')
istat = istat[istat['Regione'] == 'Lombardia']
istat

# %%
redd = pd.read_csv(DATA + 'Redditi/2011/comunali.csv', sep=';', index_col=False)
redd = redd[redd['Regione'] == 'Lombardia']
redd.head()

# %%
len(redd)

#%%
def _normalize_text(value: str) -> str:
    value = unicodedata.normalize('NFKD', str(value))
    value = value.encode('ascii', 'ignore').decode('ascii')
    return ' '.join(value.strip().lower().split())


def _normalize_join_key(value: str) -> str:
    value = unicodedata.normalize('NFKD', str(value))
    value = value.encode('ascii', 'ignore').decode('ascii').lower().strip()
    return re.sub(r'[^a-z0-9]+', '', value)


def _normalize_redditi_column_name(column: str) -> str:
    column = _normalize_text(column)
    column = column.replace('(compresi valori nulli)', '(comprensivo dei valori nulli)')
    column = column.replace('semplificata(', 'semplificata (')
    column = column.replace('ammontare in euro', 'ammontare')
    column = column.replace("contabilita'", 'contabilita')
    column = column.replace('reddito imponibile addizionale irpef', 'reddito imponibile addizionale')
    return column


def read_redditi(year: int | str, region: str = 'Lombardia') -> pd.DataFrame:
    canonical_columns = {
        'anno di imposta': 'Anno di imposta',
        'codice catastale': 'Codice catastale',
        'codice istat': 'Codice Istat Comune',
        'codice istat comune': 'Codice Istat Comune',
        'denominazione comune': 'Denominazione Comune',
        'sigla provincia': 'Sigla Provincia',
        'regione': 'Regione',
        'codice istat regione': 'Codice Istat Regione',
        'numero contribuenti': 'Numero contribuenti',
        'reddito da fabbricati - frequenza': 'Reddito da fabbricati - Frequenza',
        'reddito da fabbricati - ammontare': 'Reddito da fabbricati - Ammontare in euro',
        'reddito da lavoro dipendente e assimilati - frequenza': 'Reddito da lavoro dipendente e assimilati - Frequenza',
        'reddito da lavoro dipendente e assimilati - ammontare': 'Reddito da lavoro dipendente e assimilati - Ammontare in euro',
        'reddito da pensione - frequenza': 'Reddito da pensione - Frequenza',
        'reddito da pensione - ammontare': 'Reddito da pensione - Ammontare in euro',
        'reddito da lavoro autonomo (comprensivo dei valori nulli) - frequenza': 'Reddito da lavoro autonomo (comprensivo dei valori nulli) - Frequenza',
        'reddito da lavoro autonomo (comprensivo dei valori nulli) - ammontare': 'Reddito da lavoro autonomo (comprensivo dei valori nulli) - Ammontare in euro',
        "reddito di spettanza dell'imprenditore in contabilita ordinaria (comprensivo dei valori nulli) - frequenza": "Reddito di spettanza dell'imprenditore in contabilita ordinaria (comprensivo dei valori nulli) - Frequenza",
        "reddito di spettanza dell'imprenditore in contabilita ordinaria (comprensivo dei valori nulli) - ammontare": "Reddito di spettanza dell'imprenditore in contabilita ordinaria (comprensivo dei valori nulli) - Ammontare in euro",
        "reddito di spettanza dell'imprenditore in contabilita semplificata (comprensivo dei valori nulli) - frequenza": "Reddito di spettanza dell'imprenditore in contabilita semplificata (comprensivo dei valori nulli) - Frequenza",
        "reddito di spettanza dell'imprenditore in contabilita semplificata (comprensivo dei valori nulli) - ammontare": "Reddito di spettanza dell'imprenditore in contabilita semplificata (comprensivo dei valori nulli) - Ammontare in euro",
        'reddito da partecipazione (comprensivo dei valori nulli) - frequenza': 'Reddito da partecipazione (comprensivo dei valori nulli) - Frequenza',
        'reddito da partecipazione (comprensivo dei valori nulli) - ammontare': 'Reddito da partecipazione (comprensivo dei valori nulli) - Ammontare in euro',
        'reddito imponibile - frequenza': 'Reddito imponibile - Frequenza',
        'reddito imponibile - ammontare': 'Reddito imponibile - Ammontare in euro',
        'imposta netta - frequenza': 'Imposta netta - Frequenza',
        'imposta netta - ammontare': 'Imposta netta - Ammontare in euro',
        'bonus spettante - frequenza': 'Bonus spettante - Frequenza',
        'bonus spettante - ammontare': 'Bonus spettante - Ammontare in euro',
        'trattamento spettante - frequenza': 'Trattamento spettante - Frequenza',
        'trattamento spettante - ammontare': 'Trattamento spettante - Ammontare in euro',
        'reddito imponibile addizionale - frequenza': 'Reddito imponibile addizionale - Frequenza',
        'reddito imponibile addizionale - ammontare': 'Reddito imponibile addizionale - Ammontare in euro',
        'addizionale regionale dovuta - frequenza': 'Addizionale regionale dovuta - Frequenza',
        'addizionale regionale dovuta - ammontare': 'Addizionale regionale dovuta - Ammontare in euro',
        'addizionale comunale dovuta - frequenza': 'Addizionale comunale dovuta - Frequenza',
        'addizionale comunale dovuta - ammontare': 'Addizionale comunale dovuta - Ammontare in euro',
        'reddito complessivo minore o uguale a zero euro - frequenza': 'Reddito complessivo minore o uguale a zero euro - Frequenza',
        'reddito complessivo minore o uguale a zero euro - ammontare': 'Reddito complessivo minore o uguale a zero euro - Ammontare in euro',
        'reddito complessivo da 0 a 10000 euro - frequenza': 'Reddito complessivo da 0 a 10000 euro - Frequenza',
        'reddito complessivo da 0 a 10000 euro - ammontare': 'Reddito complessivo da 0 a 10000 euro - Ammontare in euro',
        'reddito complessivo da 10000 a 15000 euro - frequenza': 'Reddito complessivo da 10000 a 15000 euro - Frequenza',
        'reddito complessivo da 10000 a 15000 euro - ammontare': 'Reddito complessivo da 10000 a 15000 euro - Ammontare in euro',
        'reddito complessivo da 15000 a 26000 euro - frequenza': 'Reddito complessivo da 15000 a 26000 euro - Frequenza',
        'reddito complessivo da 15000 a 26000 euro - ammontare': 'Reddito complessivo da 15000 a 26000 euro - Ammontare in euro',
        'reddito complessivo da 26000 a 55000 euro - frequenza': 'Reddito complessivo da 26000 a 55000 euro - Frequenza',
        'reddito complessivo da 26000 a 55000 euro - ammontare': 'Reddito complessivo da 26000 a 55000 euro - Ammontare in euro',
        'reddito complessivo da 55000 a 75000 euro - frequenza': 'Reddito complessivo da 55000 a 75000 euro - Frequenza',
        'reddito complessivo da 55000 a 75000 euro - ammontare': 'Reddito complessivo da 55000 a 75000 euro - Ammontare in euro',
        'reddito complessivo da 75000 a 120000 euro - frequenza': 'Reddito complessivo da 75000 a 120000 euro - Frequenza',
        'reddito complessivo da 75000 a 120000 euro - ammontare': 'Reddito complessivo da 75000 a 120000 euro - Ammontare in euro',
        'reddito complessivo oltre 120000 euro - frequenza': 'Reddito complessivo oltre 120000 euro - Frequenza',
        'reddito complessivo oltre 120000 euro - ammontare': 'Reddito complessivo oltre 120000 euro - Ammontare in euro',
    }

    year = str(year)
    target_region = _normalize_text(region)
    redd = pd.read_csv(
        DATA + f'Redditi/{year}/comunali.csv',
        sep=';',
        index_col=False,
        encoding='latin1',
        low_memory=False,
        dtype={
            'Codice catastale': 'string',
            'Codice Istat': 'string',
            'Codice Istat Comune': 'string',
            'Codice Istat Regione': 'string',
            'Sigla Provincia': 'string',
            'Regione': 'string',
            'Denominazione Comune': 'string',
        },
    )

    redd = redd.drop(columns=[col for col in redd.columns if not str(col).strip()], errors='ignore')
    redd = redd.rename(
        columns={
            column: canonical_columns.get(_normalize_redditi_column_name(column), column.strip())
            for column in redd.columns
        }
    )

    for column in ['Regione', 'Denominazione Comune', 'Sigla Provincia']:
        if column in redd.columns:
            redd[column] = redd[column].astype('string').str.strip()

    if 'Regione' in redd.columns:
        redd['Regione'] = redd['Regione'].str.title()
        redd = redd[redd['Regione'].map(_normalize_text) == target_region]

    if 'Codice Istat Comune' in redd.columns:
        redd['Codice Istat Comune'] = redd['Codice Istat Comune'].str.zfill(6)

    if 'Codice Istat Regione' in redd.columns:
        redd['Codice Istat Regione'] = redd['Codice Istat Regione'].str.zfill(2)
    else:
        redd['Codice Istat Regione'] = pd.NA

    canonical_order = list(dict.fromkeys(canonical_columns.values()))
    remaining_columns = [column for column in redd.columns if column not in canonical_order]
    return redd.reindex(columns=canonical_order + remaining_columns)


def read_istat_codes(region: str = 'Lombardia') -> pd.DataFrame:
    istat = pd.read_csv(DATA + 'istat_codes.csv', sep=';', dtype=str)
    istat = istat[istat['Regione'].map(_normalize_text) == _normalize_text(region)].copy()
    istat['Codice Comune (alfanumerico)'] = istat['Codice Comune (alfanumerico)'].astype('string').str.zfill(6)
    istat['Codice Comune (numerico)'] = istat['Codice Comune (numerico)'].astype('string').str.zfill(5)
    istat['Codice catasto'] = istat['Codice catasto'].astype('string').str.upper()
    istat['Sigla automobilistica'] = istat['Sigla automobilistica'].astype('string').str.upper()
    return istat


def read_referendum2026(region: str = 'Lombardia') -> pd.DataFrame:
    ns = {'a': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with ZipFile(DATA + 'referendum2026.xlsx') as workbook:
        shared = ET.fromstring(workbook.read('xl/sharedStrings.xml'))
        shared_strings = [
            ''.join(text.text or '' for text in item.iterfind('.//a:t', ns))
            for item in shared.findall('a:si', ns)
        ]
        sheet = ET.fromstring(workbook.read('xl/worksheets/sheet1.xml'))
        rows = sheet.find('a:sheetData', ns)

        values = []
        for row in rows:
            row_values = []
            for cell in row.findall('a:c', ns):
                value = cell.find('a:v', ns)
                cell_type = cell.attrib.get('t')
                if value is None:
                    row_values.append('')
                elif cell_type == 's':
                    row_values.append(shared_strings[int(value.text)])
                else:
                    row_values.append(value.text)
            values.append(row_values)

    referendum = pd.DataFrame(values[1:], columns=values[0])
    referendum = referendum[referendum['REG'].map(_normalize_text) == _normalize_text(region)].copy()
    referendum = referendum.rename(
        columns={
            'DATAELEZIONE': 'Data elezione',
            'REG': 'Regione',
            'PROV': 'Provincia',
            'COM': 'Comune',
            'ELETTORITOTALI': 'Elettori totali',
            'NUMVOTANTITOTALI': 'Votanti totais',
            'SKBIANCHE': 'Schede bianche',
            'VOTIVALIDI': 'Voti validi',
            'VOTISI': 'Voti si',
        }
    )

    for column in ['Regione', 'Provincia', 'Comune']:
        referendum[column] = referendum[column].astype('string').str.strip()

    for column in ['Data elezione', 'Elettori totali', 'Votanti totais', 'Schede bianche', 'Voti validi', 'Voti si']:
        referendum[column] = pd.to_numeric(referendum[column], errors='coerce')

    referendum['Voti no'] = referendum['Voti validi'] - referendum['Voti si']
    referendum['Perc voti si'] = referendum['Voti si'] / referendum['Voti validi']
    referendum['Perc voti no'] = referendum['Voti no'] / referendum['Voti validi']

    referendum['_province_key'] = referendum['Provincia'].map(_normalize_join_key)
    referendum['_comune_key'] = referendum['Comune'].map(_normalize_join_key)

    istat = read_istat_codes(region=region).copy()
    istat['_province_key'] = istat['Provincia/Uts'].map(_normalize_join_key)
    istat['_comune_key'] = istat['Comune'].map(_normalize_join_key)

    referendum = referendum.merge(
        istat[
            [
                '_province_key',
                '_comune_key',
                'Codice Comune (alfanumerico)',
                'Codice Comune (numerico)',
                'Sigla automobilistica',
                'Codice catasto',
                'Provincia/Uts',
            ]
        ],
        how='left',
        on=['_province_key', '_comune_key'],
    )
    referendum = referendum.rename(
        columns={
            'Codice Comune (alfanumerico)': 'Codice Istat Comune',
            'Codice Comune (numerico)': 'Codice Istat Comune numerico',
            'Sigla automobilistica': 'Sigla Provincia',
            'Codice catasto': 'Codice catastale',
            'Provincia/Uts': 'Provincia ISTAT',
        }
    )
    return referendum.drop(columns=['_province_key', '_comune_key'])


def read_camera2022(region: str = 'Lombardia') -> pd.DataFrame:
    comune_aliases = {
        'ALBAREDO ARNABOLDI': 'CAMPOSPINOSO ALBAREDO',
        'BARDELLO': 'BARDELLO CON MALGESSO E BREGANO',
        'BREGANO': 'BARDELLO CON MALGESSO E BREGANO',
        'CAMPOSPINOSO': 'CAMPOSPINOSO ALBAREDO',
        'LIRIO': 'COLLI VERDI',
        'MALGESSO': 'BARDELLO CON MALGESSO E BREGANO',
        'RONAGO': 'UGGIATE CON RONAGO',
        'UGGIATE TREVANO': 'UGGIATE CON RONAGO',
    }
    coalition_columns = ['CENTRODESTRA', 'CENTROSINISTRA', 'M5S', 'TERZO_POLO', 'ALTRI']
    excluded_columns = {
        'COMUNE', 'COMUNE_clean', 'ELETTORITOT', 'ELETTORIM', 'VOTANTITOT', 'VOTANTIM',
        'SKBIANCHE', 'ELETTORIF', 'VOTANTIF', 'AFFLUENZA', 'AFFLUENZA_M', 'AFFLUENZA_F',
        'PERC_SCHEDE_BIANCHE', 'TOT_VOTI_LISTA', 'VINCITORE_COALIZIONE',
        'VOTI_VINCITORE_COALIZIONE', 'PERC_VINCITORE_COALIZIONE', 'VINCITORE_PARTITO',
        'VOTI_VINCITORE_PARTITO', 'PERC_VINCITORE_PARTITO', 'POLARIZZAZIONE_DX_SX',
        'CHECK_COALIZIONI',
    }

    camera = pd.read_csv(DATA + 'camera2022.csv')
    camera['Comune'] = camera['COMUNE_clean'].fillna(camera['COMUNE']).astype('string').str.strip()
    camera['Comune ISTAT lookup'] = camera['Comune'].replace(comune_aliases)
    camera['_comune_key'] = camera['Comune ISTAT lookup'].map(_normalize_join_key)

    for column in camera.columns:
        if column not in {'COMUNE', 'COMUNE_clean', 'Comune', 'Comune ISTAT lookup', '_comune_key'}:
            try:
                camera[column] = pd.to_numeric(camera[column])
            except (ValueError, TypeError):
                pass

    istat = read_istat_codes(region=region).copy()
    istat['_comune_key'] = istat['Comune'].map(_normalize_join_key)
    camera = camera.merge(
        istat[['_comune_key', 'Codice Comune (alfanumerico)', 'Codice catasto']],
        how='left',
        on='_comune_key',
    )
    camera = camera.rename(
        columns={
            'Codice Comune (alfanumerico)': 'Codice Istat Comune',
            'Codice catasto': 'Codice catastale',
        }
    )

    party_columns = [
        column for column in camera.columns
        if column not in excluded_columns
        and column not in coalition_columns
        and not column.endswith('_PERC')
        and column not in {'Comune', 'Comune ISTAT lookup', '_comune_key', 'Codice Istat Comune', 'Codice catastale'}
    ]
    renamed_columns = {column: f'Voti partito - {column}' for column in party_columns}
    renamed_columns.update({column: f'Voti coalizione - {column}' for column in coalition_columns})
    camera = camera.rename(columns=renamed_columns)

    keep_columns = (
        ['Comune', 'Codice Istat Comune', 'Codice catastale', 'VOTANTITOT']
        + [renamed_columns[column] for column in coalition_columns]
        + [renamed_columns[column] for column in party_columns]
    )
    camera = camera.rename(columns={'VOTANTITOT': 'Votanti totali politiche 2022'})
    keep_columns[3] = 'Votanti totali politiche 2022'
    return camera[keep_columns]

#%%
def read_omi(year: int | str, region: str = 'Lombardia') -> pd.DataFrame:
    canonical_columns = [
        'Area territoriale',
        'Regione',
        'Sigla Provincia',
        'Codice ISTAT Comune',
        'Codice Catastale Comune',
        'Sezione',
        'Codice Amministrativo Comune',
        'Comune',
        'Fascia',
        'Zona',
        'Link Zona',
        'Codice Tipologia',
        'Tipologia',
        'Stato conservativo',
        'Stato conservativo prevalente',
        'Prezzo minimo di compravendita',
        'Prezzo massimo di compravendita',
        'Superficie non locale compravendita',
        'Canone minimo di locazione',
        'Canone massimo di locazione',
        'Superficie non locale locazione',
    ]
    renamed_columns = {
        'Area_territoriale': 'Area territoriale',
        'Prov': 'Sigla Provincia',
        'Comune_ISTAT': 'Codice ISTAT Comune',
        'Comune_cat': 'Codice Catastale Comune',
        'Sez': 'Sezione',
        'Comune_amm': 'Codice Amministrativo Comune',
        'Comune_descrizione': 'Comune',
        'LinkZona': 'Link Zona',
        'Cod_Tip': 'Codice Tipologia',
        'Descr_Tipologia': 'Tipologia',
        'Stato': 'Stato conservativo',
        'Stato_prev': 'Stato conservativo prevalente',
        'Compr_min': 'Prezzo minimo di compravendita',
        'Compr_max': 'Prezzo massimo di compravendita',
        'Sup_NL_compr': 'Superficie non locale compravendita',
        'Loc_min': 'Canone minimo di locazione',
        'Loc_max': 'Canone massimo di locazione',
        'Sup_NL_loc': 'Superficie non locale locazione',
    }

    year = str(year)
    target_region = _normalize_text(region)
    omi = pd.read_csv(
        DATA + f'omi/{year}/quotazioni.csv',
        sep=';',
        encoding='latin1',
        low_memory=False,
        dtype={
            'Area_territoriale': 'string',
            'Regione': 'string',
            'Prov': 'string',
            'Comune_ISTAT': 'string',
            'Comune_cat': 'string',
            'Sez': 'string',
            'Comune_amm': 'string',
            'Comune_descrizione': 'string',
            'Fascia': 'string',
            'Zona': 'string',
            'LinkZona': 'string',
            'Cod_Tip': 'string',
            'Descr_Tipologia': 'string',
            'Stato': 'string',
            'Stato_prev': 'string',
            'Sup_NL_compr': 'string',
            'Sup_NL_loc': 'string',
        },
    )

    omi = omi.drop(columns=[col for col in omi.columns if not str(col).strip() or str(col).startswith('Unnamed:')], errors='ignore')
    omi = omi.rename(columns=renamed_columns)

    for column in [
        'Area territoriale',
        'Regione',
        'Sigla Provincia',
        'Codice ISTAT Comune',
        'Codice Catastale Comune',
        'Sezione',
        'Codice Amministrativo Comune',
        'Comune',
        'Fascia',
        'Zona',
        'Link Zona',
        'Codice Tipologia',
        'Tipologia',
        'Stato conservativo',
        'Stato conservativo prevalente',
        'Superficie non locale compravendita',
        'Superficie non locale locazione',
    ]:
        if column in omi.columns:
            omi[column] = omi[column].astype('string').str.strip()

    if 'Regione' in omi.columns:
        omi['Regione'] = omi['Regione'].str.title()
        omi = omi[omi['Regione'].map(_normalize_text) == target_region]

    if 'Codice ISTAT Comune' in omi.columns:
        omi['Codice ISTAT Comune'] = omi['Codice ISTAT Comune'].str.zfill(7)

    for column in ['Codice Catastale Comune', 'Codice Amministrativo Comune', 'Sigla Provincia']:
        if column in omi.columns:
            omi[column] = omi[column].str.upper()

    for column in [
        'Prezzo minimo di compravendita',
        'Prezzo massimo di compravendita',
        'Canone minimo di locazione',
        'Canone massimo di locazione',
    ]:
        if column in omi.columns:
            omi[column] = pd.to_numeric(omi[column].astype('string').str.replace(',', '.', regex=False), errors='coerce')

    remaining_columns = [column for column in omi.columns if column not in canonical_columns]
    return omi.reindex(columns=canonical_columns + remaining_columns)


def read_geodata(region: str = 'Lombardia') -> gpd.GeoDataFrame:
    renamed_columns = {
        'nome_reg': 'Regione',
        'sig_pro': 'Sigla Provincia',
        'nome_pro': 'Provincia',
        'nome_com': 'Comune',
        'belfiore': 'Codice catastale',
        'cod_istatn': 'Codice ISTAT Comune esteso',
        'istat': 'Codice ISTAT Comune',
    }

    geo = gpd.read_file(DATA + 'limiti_comunali_2020.geojson')
    geo = geo.drop(columns='anno', errors='ignore')
    geo = geo.rename(columns=renamed_columns)

    for column in ['Regione', 'Sigla Provincia', 'Provincia', 'Comune', 'Codice catastale']:
        if column in geo.columns:
            geo[column] = geo[column].astype('string').str.strip()

    if 'Regione' in geo.columns:
        geo['Regione'] = geo['Regione'].str.title()
        geo = geo[geo['Regione'].map(_normalize_text) == _normalize_text(region)]

    if 'Codice catastale' in geo.columns:
        geo['Codice catastale'] = geo['Codice catastale'].str.upper()

    if 'Codice ISTAT Comune' in geo.columns:
        geo['Codice ISTAT Comune'] = geo['Codice ISTAT Comune'].astype('string').str.zfill(6)

    if 'Codice ISTAT Comune esteso' in geo.columns:
        geo['Codice ISTAT Comune esteso'] = geo['Codice ISTAT Comune esteso'].astype('string').str.zfill(8)

    return geo


def _normalize_years(years: int | str | list | tuple | set) -> list[str]:
    if isinstance(years, (list, tuple, set)):
        return [str(year) for year in years]
    return [str(years)]


def _belfiore(series: pd.Series) -> pd.Series:
    return series.astype('string').str.strip().str.upper()


def _prepare_redditi_for_merge(region: str = 'Lombardia') -> pd.DataFrame:
    redd = read_redditi(year=2021, region=region).copy()
    redd['avg_income'] = redd['Reddito imponibile - Ammontare in euro'] / redd['Numero contribuenti']
    redd['belfiore'] = _belfiore(redd['Codice catastale'])
    return redd


def _prepare_omi_for_merge(region: str = 'Lombardia') -> pd.DataFrame:
    omi = read_omi(year=2021, region=region).copy()
    omi['price_m2'] = omi[['Prezzo minimo di compravendita', 'Prezzo massimo di compravendita']].mean(axis=1)
    omi['rent_m2'] = omi[['Canone minimo di locazione', 'Canone massimo di locazione']].mean(axis=1)
    omi['belfiore'] = _belfiore(omi['Codice Amministrativo Comune'])
    omi_muni = (
        omi.groupby('belfiore', dropna=False)
        .agg(
            price_m2=('price_m2', 'mean'),
            rent_m2=('rent_m2', 'mean'),
            n_quotes=('price_m2', 'size'),
        )
        .reset_index()
    )
    return omi_muni


def merge_geodata_omi_redditi_2021(region: str = 'Lombardia') -> gpd.GeoDataFrame:
    geo = read_geodata(region=region).copy()
    geo['belfiore'] = _belfiore(geo['Codice catastale'])
    redd = _prepare_redditi_for_merge(region=region)
    omi_muni = _prepare_omi_for_merge(region=region)

    merged = geo.merge(
        redd[
            [
                'belfiore',
                'Denominazione Comune',
                'Sigla Provincia',
                'Numero contribuenti',
                'avg_income',
                'Reddito imponibile - Ammontare in euro',
                'Anno di imposta',
                'Codice Istat Comune',
                'Codice Istat Regione',
            ]
        ],
        on='belfiore',
        how='left',
        suffixes=('_geo', '_redditi'),
    )
    merged = merged.merge(omi_muni, on='belfiore', how='left')
    merged['Anno'] = 2021
    merged['OMI Anno'] = 2021
    return gpd.GeoDataFrame(merged, geometry='geometry', crs=geo.crs)


def merge_geodata_omi_redditi(
    year: int | str | list | tuple | set = 2021,
    region: str = 'Lombardia',
    omi_year: int | str | None = None,
) -> gpd.GeoDataFrame:
    year_values = _normalize_years(year)
    omi_year_value = str(omi_year) if omi_year is not None else '2021'
    if year_values != ['2021'] or omi_year_value != '2021':
        raise ValueError('Esta função agora replica o notebook apenas para 2021. Use year=2021 e omi_year=2021.')
    return merge_geodata_omi_redditi_2021(region=region)


def merge_referendum2026_with_merged_df(
    merged_df: pd.DataFrame | gpd.GeoDataFrame,
    region: str = 'Lombardia',
) -> pd.DataFrame | gpd.GeoDataFrame:
    referendum = read_referendum2026(region=region).copy()
    merged = merged_df.copy()

    if 'Codice ISTAT Comune_geo' in merged.columns:
        merge_key = 'Codice ISTAT Comune_geo'
    elif 'Codice Istat Comune' in merged.columns:
        merge_key = 'Codice Istat Comune'
    elif 'Codice ISTAT Comune' in merged.columns:
        merge_key = 'Codice ISTAT Comune'
    else:
        raise KeyError(
            "merged_df precisa ter uma das colunas 'Codice ISTAT Comune_geo', "
            "'Codice Istat Comune' ou 'Codice ISTAT Comune' para o merge com referendum2026."
        )

    merged[merge_key] = merged[merge_key].astype('string').str.zfill(6)
    referendum['Codice Istat Comune'] = referendum['Codice Istat Comune'].astype('string').str.zfill(6)

    output = merged.merge(
        referendum[
            [
                'Codice Istat Comune',
                'Perc voti si',
                'Perc voti no',
            ]
        ],
        how='left',
        left_on=merge_key,
        right_on='Codice Istat Comune',
        suffixes=('', '_referendum'),
    )

    if isinstance(merged_df, gpd.GeoDataFrame):
        return gpd.GeoDataFrame(output, geometry=merged_df.geometry.name, crs=merged_df.crs)
    return output


def merge_geodata_omi_redditi_referendum2026(
    year: int | str | list | tuple | set,
    region: str = 'Lombardia',
) -> pd.DataFrame | gpd.GeoDataFrame:
    merged_df = merge_geodata_omi_redditi(year=year, region=region)
    return merge_referendum2026_with_merged_df(merged_df=merged_df, region=region)


def merge_camera2022_with_merged_df(
    merged_df: pd.DataFrame | gpd.GeoDataFrame,
    region: str = 'Lombardia',
) -> pd.DataFrame | gpd.GeoDataFrame:
    camera = read_camera2022(region=region).copy()
    merged = merged_df.copy()

    if 'Codice Istat Comune' in merged.columns:
        merge_key = 'Codice Istat Comune'
    elif 'Codice ISTAT Comune' in merged.columns:
        merge_key = 'Codice ISTAT Comune'
    else:
        raise KeyError(
            "merged_df precisa ter uma das colunas 'Codice Istat Comune' ou 'Codice ISTAT Comune' para o merge com camera2022."
        )

    merged[merge_key] = merged[merge_key].astype('string').str.zfill(6)
    camera['Codice Istat Comune'] = camera['Codice Istat Comune'].astype('string').str.zfill(6)
    output = merged.merge(
        camera,
        how='left',
        left_on=merge_key,
        right_on='Codice Istat Comune',
        suffixes=('', '_camera2022'),
    )

    if isinstance(merged_df, gpd.GeoDataFrame):
        return gpd.GeoDataFrame(output, geometry=merged_df.geometry.name, crs=merged_df.crs)
    return output


def merge_geodata_omi_redditi_votes_2021(region: str = 'Lombardia') -> pd.DataFrame | gpd.GeoDataFrame:
    merged_df = merge_geodata_omi_redditi_2021(region=region)
    merged_df = merge_referendum2026_with_merged_df(merged_df=merged_df, region=region)
    return merge_camera2022_with_merged_df(merged_df=merged_df, region=region)

# %%
def concat_redditi(years):
    dfs = [read_redditi(year) for year in years]
    return pd.concat(dfs, axis=0)

def concat_omi(years):
    dfs = [read_omi(year) for year in years]
    return pd.concat(dfs, axis=0)

# %%
df = merge_geodata_omi_redditi_votes_2021()
df.head()

# %%
ax = df.plot(
    column='avg_income',
    cmap='Blues',
    figsize=(10, 10),
    legend=True,
    edgecolor='white',
    linewidth=0.2,
    missing_kwds={'color': '#f2f2f2'}
)
ax.set_axis_off()
plt.tight_layout()

# %%
