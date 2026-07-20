from tqdm import tqdm

tqdm.pandas()

from ai_population.config.market_signals_config import SEARCH_TERMS_X
from ai_population.src.utils import (
    perform_x_profile_metadata_search,
    perform_x_profile_search,
    perform_x_keyword_search,
)


## LADT Data Collection Configurations
# PROJECT_NAME_X = "ladt-data-collection"
# PIPELINE_EXECUTION_DATE = "20-06-2026"
# POOL_FILE_X = "ladt-profile-pool-x.csv"
# PROFILE_METADATA_SEARCH_FILE_X = (
#     f"x_ladt_profile_metadata_{PIPELINE_EXECUTION_DATE}.csv"
# )
# PROFILE_SEARCH_FILE_X = f"x_ladt_profile_search_{PIPELINE_EXECUTION_DATE}.csv"
# PROFILE_SEARCH_START_DATE = "2026-01-01"
# PROFILE_SEARCH_END_DATE = "2026-06-10"
# NUM_POSTS_PER_PROFILE = 100

## Digital Twin Chile Data Collection Configurations
# PROJECT_NAME_X = "digital-twin-chile-x-round2"
# PIPELINE_EXECUTION_DATE = "23-06-2026"
# POOL_FILE_X = "digital-twin-chile-profile-pool-x.csv"
# PROFILE_METADATA_SEARCH_FILE_X = (
#     f"x_digital_twin_chile_round2_profile_metadata_{PIPELINE_EXECUTION_DATE}.csv"
# )
# PROFILE_SEARCH_FILE_X = f"x_digital_twin_chile_round2_profile_search_{PIPELINE_EXECUTION_DATE}.csv"
# PROFILE_SEARCH_START_DATE = "2025-08-29"
# PROFILE_SEARCH_END_DATE = "2026-06-23"
# NUM_POSTS_PER_PROFILE = 250

## Swiss Election Keyword Configurations
PROJECT_NAME_X = "swiss-election-keyword-search-analysis-x"
PIPELINE_EXECUTION_DATE = "23-06-2026"
SEARCH_TERMS_X = [
    "Schweizer Bundesrat",
    "Bundesrat Schweiz",
    "Bundesrätin Schweiz",
    "Nationalrat Schweiz",
    "Nationalrätin Schweiz",
    "Ständerat Schweiz",
    "Ständerätin Schweiz",
    "Bundeshaus Bern",
    "Abstimmungssonntag",
    "Volksabstimmung Schweiz",
    "Volksinitiative Schweiz",
    "Schweizer Referendum",
    "Schweizer Initiative",
    "direkte Demokratie Schweiz",
    "Föderalismus Schweiz",
    "Eidgenossenschaft",
    "Schweizerische Volkspartei",
    "SVP Schweiz",
    "Sozialdemokratische Partei der Schweiz",
    "SP Schweiz",
    "FDP Schweiz",
    "FDP.Die Liberalen",
    "Grüne Schweiz",
    "GLP Schweiz",
    "Grünliberale Schweiz",
    "Die Mitte Schweiz",
    "EVP Schweiz",
    "Nationalratswahlen Schweiz",
    "Schweizer Neutralität",
    "Bilaterale Abkommen Schweiz",
    "Bilaterale III Schweiz",
    "Asylpolitik Schweiz",
    "Klimapolitik Schweiz",
    "Conseil fédéral suisse",
    "Conseillère fédérale suisse",
    "Conseil national suisse",
    "Conseillère nationale suisse",
    "Conseil des États suisse",
    "Assemblée fédérale suisse",
    "Palais fédéral suisse",
    "votation suisse",
    "initiative populaire suisse",
    "référendum suisse",
    "démocratie directe suisse",
    "dimanche de vote suisse",
    "Confédération suisse",
    "UDC Suisse",
    "Parti socialiste suisse",
    "PS Suisse",
    "PLR Suisse",
    "Parti libéral-radical suisse",
    "Verts Suisse",
    "PVL Suisse",
    "Le Centre Suisse",
    "élections fédérales suisse",
    "neutralité suisse",
    "accords bilatéraux suisse",
    "accords bilatéraux III suisse",
    "politique d'asile suisse",
    "politique climatique suisse",
    "Consiglio federale svizzero",
    "Consigliera federale svizzera",
    "Consiglio nazionale svizzero",
    "Consiglio degli Stati svizzero",
    "Assemblea federale svizzera",
    "Palazzo federale svizzero",
    "votazione svizzera",
    "iniziativa popolare svizzera",
    "referendum svizzero",
    "democrazia diretta Svizzera",
    "Confederazione Svizzera",
    "UDC Svizzera",
    "Partito socialista svizzero",
    "PS Svizzera",
    "PLR Svizzera",
    "Partito liberale-radicale svizzero",
    "Verdi Svizzera",
    "PVL Svizzera",
    "Il Centro Svizzera",
    "Lega dei Ticinesi",
    "elezioni federali svizzera",
    "neutralità svizzera",
    "accordi bilaterali svizzera",
    "accordi bilaterali III svizzera",
    "politica d'asilo svizzera",
    "politica climatica svizzera",
]
NUM_POSTS_PER_KEYWORD = 50
KEYWORD_SEARCH_FILE_X = "swiss-election-keyword-search-analysis-x.csv"

if __name__ == "__main__":
    # perform_x_profile_metadata_search(
    #     project_name=PROJECT_NAME_X,
    #     execution_date=PIPELINE_EXECUTION_DATE,
    #     input_file=POOL_FILE_X,
    #     output_file=PROFILE_METADATA_SEARCH_FILE_X,
    # )
    # perform_x_profile_search(
    #     project_name=PROJECT_NAME_X,
    #     execution_date=PIPELINE_EXECUTION_DATE,
    #     input_file=POOL_FILE_X,
    #     output_file=PROFILE_SEARCH_FILE_X,
    #     start_date=PROFILE_SEARCH_START_DATE,
    #     end_date=PROFILE_SEARCH_END_DATE,
    #     num_posts_per_profile=NUM_POSTS_PER_PROFILE,
    # )
    perform_x_keyword_search(
        project_name=PROJECT_NAME_X,
        execution_date=PIPELINE_EXECUTION_DATE,
        search_terms=SEARCH_TERMS_X,
        output_file=KEYWORD_SEARCH_FILE_X,
        num_posts_per_keyword=NUM_POSTS_PER_KEYWORD,
    )
