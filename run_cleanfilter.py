"""
This script is responsible for updating all tables that rely on the dishes table. The script is designed to be executed daily after the run_scraper.py as a cronjob. All the gtp
promts are imported from the scraper.gpt_prompts module. The script is divided into several modules, each responsible for updating a specific table.


Modules:
- update_filters_clean loads the menu and icon column from the dishes table which are not yet included in the CleanFilters table. Then it cleans the strings in the icon column and updates the CleanFilters table with the cleaned strings.
- update_Course(), update_embeddings(), update_description(), update_taste(), update_recipe(), update_ingredients(), write_to_dishes_eng() load the menu column that is not contained in the respective table yet and applies the prompts.
- update_cleajnprices() loads the dishes table and imputes the missing prices for the studentPrice and guestPrice columns. The imputed prices are then written back to the CleanFilters table.

Execution:
This script is intended to be run as a cronjob every day at 3:30 AM.
"""

from db.db_read import setup_database_connection
from secret import USER, PASSWORD, HOST, PORT
import pandas as pd
import numpy as np
from scraper.gemini_prompts import classify_missing_filters,embedding_extraction,generate_description,classify_dish_taste,generate_recipe,ingredient_extraction
from db.db_read import load_dishes_table_for_filter_cleaning, get_combined_dishes
from db.utils import correct_icons,impute_missing_prices
from db.db_write import write_to_filters_clean,write_to_course,write_to_embedding, write_to_description, write_to_taste, write_to_recipe, write_to_ingredient,write_to_dishes_eng,write_imputed_price_to_filtersclean



replacement_dict = {
            "F": "fish",
            "G": "poultry",
            "K": "veal",
            "L": "lamb",
            "R": "beef",
            
            "S": "pork",
            "W": "game",
            "V": "vegetarian"
        }

engine, Session = setup_database_connection(USER, PASSWORD, HOST, PORT)


# replace categories
def replace_categories(icon_string):
    if pd.isna(icon_string):
        return None
    # Split the string into parts, map replacements, and join back
    return ", ".join(replacement_dict.get(part.strip(), part.strip()) for part in icon_string.split(","))


def update_FiltersClean():

    # load dishes where the icon column in FiltersClean table is not updated yet.
    dish = load_dishes_table_for_filter_cleaning(Session,"FiltersClean")

    # Stop the function if 'dish' is empty
    if dish.empty:
        return

    dish.replace("NA", np.nan, inplace=True)

    # Apply the function to the DataFrame
    dish["icons"] = dish["icons"].apply(replace_categories)

    # vegan labels tend to be double 'Vegan, vegan'
    dish["icons"] = dish["icons"].apply(lambda x: "vegan" if "vegan" in str(x).lower() else x)

    # correct the missing filters by hand as the dishes belong to one category quite obviously
    dish['icons_clean'] = np.where(
        dish['icons'].isna(),  # Condition: Check where 'icons' is NaN
        dish.apply(lambda row: correct_icons(row['menu'], row['menuLine']), axis=1),  
        dish['icons']  
    )

    # If main dishes are still not classified, retrieve the classification from chatgpt.
    filtered_menu = ["Angebot des Tages", "Auswahlgericht", "Tagesmenü","Auswahlgericht 2"]
    dish.loc[
        dish['icons_clean'].isna() & dish['menuLine'].isin(filtered_menu), 
        'icons_clean'
    ] = dish.loc[
        dish['icons_clean'].isna() & dish['menuLine'].isin(filtered_menu), 
        'menu'
    ].apply(classify_missing_filters)

    with Session() as db_session:
        write_to_filters_clean(dish,engine,db_session)
    print("FiltersClean is updated")

def update_Course():
    with Session() as db_session:
        write_to_course(engine,db_session)
    print("Course is updated")

def update_embeddings():
    df = load_dishes_table_for_filter_cleaning(Session,"Embedding")

    if df.empty:
        return

    with Session() as db_session:
        embeddings_df = embedding_extraction(df, 'menu')
        write_to_embedding(embeddings_df, engine, db_session)
    print("Embedding is updated")

def update_description():
    df = load_dishes_table_for_filter_cleaning(Session,"Description")

    if df.empty:
        return

    with Session() as db_session:
        description_df = generate_description(df, 'menu','menuLine')
        write_to_description(description_df,engine,db_session)
    print("Description is updated")

def update_taste():
    df = load_dishes_table_for_filter_cleaning(Session,"Taste")

    if df.empty:
        return
    
    with Session() as db_session:
        taste_df = classify_dish_taste(df, 'menu')
        write_to_taste(taste_df, engine,db_session)
    print("Taste is updated")

def update_recipe():
    df = load_dishes_table_for_filter_cleaning(Session,"Recipe")

    if df.empty:
        return
    
    with Session() as db_session:
        recipe_df = generate_recipe(df, 'menu', 'menuLine')
        write_to_recipe(recipe_df, engine,db_session)
    print("Recipe is updated")

def update_ingredients():
    df = load_dishes_table_for_filter_cleaning(Session,"Ingredient")

    if df.empty:
        return
    
    with Session() as db_session:
        ingredients_df = ingredient_extraction(df, 'menu')
        write_to_ingredient(ingredients_df, engine,db_session)
    print("Ingredients is updated")

def update_cleanprices():


    with Session() as db_session:
        dish = get_combined_dishes(db_session)

        if dish.empty:
            return
        
        impute_studentPrice = impute_missing_prices(dish,"studentPrice")
        impute_guestPrice = impute_missing_prices(impute_studentPrice,"guestPrice")
        write_imputed_price_to_filtersclean(impute_studentPrice,db_session,"studentPrice",engine)
        write_imputed_price_to_filtersclean(impute_guestPrice,db_session,"guestPrice",engine)
    print("Prices are updated")

if __name__ == "__main__":
    update_FiltersClean()
    update_Course()
    update_embeddings()
    update_description()
    update_taste()
    update_recipe()
    update_ingredients()
    write_to_dishes_eng(engine,Session)
    update_cleanprices()
