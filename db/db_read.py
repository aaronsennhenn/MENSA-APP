"""
This Script containts all functions that read from the database
"""

from sqlalchemy.orm import Session
from .db_write import Dish, Directory, setup_database_connection, User, Rating, Course, Description, Recipe, Embedding, FiltersClean, DishEng, Taste, Ingredient, DishHistory,PriceClean,EmbeddingCluster
from typing import List
from sqlalchemy import func
import datetime
import pandas as pd
from sqlalchemy import or_,func,select,desc,select, union, and_, not_, extract
from typing import List, Tuple
from db.utils import compute_cosine_similarity, get_month_name,format_price_column
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from datetime import datetime, timedelta



def convert_dblist_to_df(db_list: List) -> pd.DataFrame:
    """
    Converts a list of database objects into a Pandas DataFrame.

    Parameters:
        db_list (list): A list of database objects, typically ORM instances (e.g., SQLAlchemy objects).

    Returns:
        pd.DataFrame: A DataFrame containing the attributes of the objects in db_list as columns, with one row per object.
    """
    dict_list = [vars(obj) for obj in db_list]
    for dictionary in dict_list:
        dictionary.pop('_sa_instance_state', None)
    df = pd.DataFrame(dict_list)        
    return df
        
def get_user_by_username(db_session: Session, username: str):
    """
    Retrieves row from the User table with the given username.

    Parameters:
        db_session (Session): The database session used to query the database.
        username (str): The username of the user to retrieve.

    Returns:
        User: The user object corresponding to the given username, or None if no user is found.

    """
    return db_session.query(User).filter_by(username=username).first()


def get_all_dishes(session: Session) -> List[Dish]:
    """
    Retrieve all dishes from the database.

    Parameters:
        session (Session): The database session to use for the query.

    Returns:
        List[Dish]: A list of all Dish objects in the database.

    """
    return session.query(Dish).all()

def get_average_ratings(db_session: Session, user_name=None) -> dict:
    """
    Get the average rating for each icon category in the database.

    Parameters:
        db_session (Session): The database session to use for the query.
        user_name (str): The name of the user to filter the ratings by. If None, all ratings are used.

    Returns:
        dict: A dictionary mapping each category to its average rating.
    """

    # Query FiltersClean and join Ratings
    query = db_session.query(
        FiltersClean.icons_clean,
        Rating.rating
    ).join(Rating, FiltersClean.menu_id == Rating.menu_id)

    # if user_name is provided, filter by username
    if user_name:
        query = query.filter(Rating.user_name == user_name)

    raw_ratings = query.all()

    # account for two categories inside one string
    category_ratings = defaultdict(list)

    for icons, rating in raw_ratings:
        categories = icons.split(", ")
        for category in categories:
            category_ratings[category].append(rating)

    # Compute the average rating for each category
    average_ratings = {
        category: sum(ratings) / len(ratings) for category, ratings in category_ratings.items()
    }

    return average_ratings

def get_dishes_by_date_location_filtered(db_session:Session, date: str, mensa_name: str, selected_diet_meat: List[str], selected_lang: str) -> List[Tuple]:
    """
    Get all dishes for a given date and mensa, with optional filtering by diet and meat icons.

    Parameters:
        db_session (Session): The database session to use for the query.
        date (str): The date to filter by, in the format "YYYY-MM-DD".
        mensa_name (str): The name of the mensa to filter by, or "all" to include all mensas.
        selected_diet_meat (List[str]): A list of diet and meat icons to filter by.
        selected_lang (str): The language to use for the menu and course fields ("de" or "en").

    Returns:
        List[Tuple]: A list of tuples containing the columns of the Dish table for the filtered dishes.
    """

    if mensa_name == "all":
        query = db_session.query(Dish).filter(Dish.menuDate == date, Dish.menu != "NA")
    else:
        query = db_session.query(Dish).filter(Dish.menuDate == date, Dish.location == mensa_name, Dish.menu != "NA")

    # join icons_clean column for filtering
    query = query.outerjoin(FiltersClean, Dish.id == FiltersClean.menu_id).add_columns(FiltersClean.icons_clean.label("icons_clean"))

    # if user filters by meat or diet, apply filters on icons_clean column
    if selected_diet_meat:
        filters = [FiltersClean.icons_clean.ilike(f"%{icon}%") for icon in selected_diet_meat]
        query = query.filter(or_(*filters))

    # Join additional tables and add columns
    query = (
        query
        .outerjoin(Recipe, Dish.id == Recipe.menu_id)
        .outerjoin(Description, Dish.id == Description.menu_id)
        .outerjoin(DishEng, Dish.id == DishEng.menu_id)
        .outerjoin(Course, Dish.id == Course.menu_id)
        .add_columns(
            # German columns
            Recipe.recipe_de,
            Description.description_de,
            Course.course,
            
            # English columns
            DishEng.menuLineEng if selected_lang == "en" else None,
            DishEng.menuEng if selected_lang == "en" else None,
            Description.description_en if selected_lang == "en" else None,
            Recipe.recipe_en if selected_lang == "en" else None,
            Course.course_eng if selected_lang == "en" else None,

        )
    )

    # join the embeddings column 
    query = query.outerjoin(Embedding, Dish.id == Embedding.menu_id).add_columns(Embedding.embedding)

    # Compute average rating for each dish
    avg_rating_subquery = (
        db_session.query(
            Rating.menu_id.label("menu_id"),
            func.round(func.avg(Rating.rating),2).label("average_rating"),
            func.count(Rating.rating).label("rating_count")  # Count of ratings

        )
        .group_by(Rating.menu_id)
        .subquery()
    )

    # Join the average rating to the main query
    query = query.outerjoin(avg_rating_subquery, Dish.id == avg_rating_subquery.c.menu_id).add_columns(
        avg_rating_subquery.c.average_rating.label("average_rating"),
        avg_rating_subquery.c.rating_count.label("rating_count")

    )

    # join the imputed prices
    query = query.outerjoin(PriceClean, Dish.id == PriceClean.menu_id).add_columns(PriceClean.studentPrice_imputed,PriceClean.guestPrice_imputed)

    return query.all()

def get_random_dishes(selected_date: datetime.date, lang: str, user_name: str, db_session: Session):
    """
    Retrieves one random dish that has not been rated by the user
    and is not part of the menu on the given date.

    Parameters:
        selected_date (datetime.date): The date to exclude dishes from.
        lang (str): The language preference ('de' for German, any other value for English).
        user_name (str): The username of the user for whom to filter out rated dishes.
        db_session (Session): The SQLAlchemy session used to execute the query.

    Returns:
        Row: A single database row representing the random dish
    """

    # Query one random main dish that has not been rated by the user yet and is not on the selected date
    if lang == "de":
        query = (
            select(Dish.id, Dish.menu, Dish.studentPrice,Dish.guestPrice)
            .outerjoin(Rating, (Dish.id == Rating.menu_id) & (Rating.user_name == user_name))
            .outerjoin(Course, Course.menu_id == Dish.id)
            .where(Rating.menu_id == None,
                Dish.menuDate != selected_date,
                Course.course == "Hauptspeise",
                Dish.menu != "NA",
                Dish.studentPrice > 0,
                Dish.guestPrice > 0
            )
            .order_by(func.random())
            .limit(1)
        )
    else:
        # English version
        query = (
            select(DishEng.menu_id, DishEng.menuEng,Dish.studentPrice,Dish.guestPrice)
            .outerjoin(Dish, Dish.id == DishEng.menu_id)
            .outerjoin(Rating, (DishEng.menu_id == Rating.menu_id) & (Rating.user_name == user_name))
            .outerjoin(Course, Course.menu_id == DishEng.menu_id)
            .where(Rating.menu_id == None,
                DishEng.menuDate != selected_date,
                Course.course_eng == "Main Dish",
                DishEng.menuEng != "N/a",
                Dish.studentPrice > 0,
                Dish.guestPrice > 0
            )
            .order_by(func.random())
            .limit(1)
        )

    return db_session.execute(query).fetchone()

def get_user_vector(username: str, session: Session) -> List[float]:
    """
    Retrieves the user vector for a specific user.

    Parameters:
        username (str): The username of the user whose vector is to be retrieved.
        session (Session): The SQLAlchemy session used to query the database.

    Returns:
        List[float]: The user's vector as a list of floats
    """
    user = session.query(User).filter_by(username=username).first()
    return user.user_vector if user else None

def load_dishes_table_for_filter_cleaning(Session: Session, update_table: str) -> pd.DataFrame:
    """
    Loads dishes from the provided table that are not present in the specified update table. Is used for the cronjob to update all table daily that depend on the Dish table.

    Parameters:
        Session (Session): The SQLAlchemy session factory used to query the database.
        update_table (str): The name of the table to update 

    Returns:
        pandas.DataFrame: A DataFrame containing the dishes that are not present in the specified update table.
    """
    if update_table == "FiltersClean":
        table = FiltersClean
    elif update_table == "Embedding":
        table = Embedding
    elif update_table == "Description":
        table = Description
    elif update_table == "Taste":
        table = Taste
    elif update_table == "Recipe":
        table = Recipe
    elif update_table == "Ingredient":
        table = Ingredient


    with Session() as session:
        # Query for dishes not in update_table
        dishes = session.query(
            Dish.id,
            Dish.menuLine,
            Dish.menu,
            Dish.icons
        ).filter(
            ~session.query(table).filter(table.menu_id == Dish.id).exists()
        ).all()

        # Convert results to a list of dictionaries
        data = [
            {
                'id': dish.id,
                'menuLine': dish.menuLine,
                'menu': dish.menu,
                'icons': dish.icons
            }
            for dish in dishes
        ]
        
        # Return as a DataFrame
        return pd.DataFrame(data)

def get_unique_mensas(session: Session) -> List[str]:
    """
    Retrieves a list of unique mensa locations from the database.

    Parameters:
        session (Session): The SQLAlchemy session used to query the database.

    Returns:
        List[str]: A list of unique mensa locations as strings.
    """
    return [mensa.location for mensa in session.query(Dish.location).distinct()]

def get_total_mensas(session: Session) -> int:
    """
    Calculates the total number of unique mensa locations in the database.

    Parameters:
        session (Session): The SQLAlchemy session used to query the database.

    Returns:
        int: The total number of distinct mensa locations.
    """
    return session.query(Dish.location).distinct().count()

def get_total_ratings(session: Session) -> int:
    """
    Calculates the total number of unique ratings in the database.

    Parameters:
        session (Session): The SQLAlchemy session used to query the database.

    Returns:
        int: The total count of distinct ratings.
    """
    return session.query(Rating.id).distinct().count()

def get_total_menus(session: Session) -> int:
    """
    Calculates the total number of unique menus in the database.

    Parameters:
        session (Session): The SQLAlchemy session used to query the database.

    Returns:
        int: The total count of distinct menus.
    """
    return session.query(Dish.id).distinct().count()

def get_first_updated_date(session: Session) -> datetime:
    """
    Retrieves the earliest dish date from the database.

    Parameters:
        session (Session): The SQLAlchemy session used to query the database.

    Returns:
        datetime: The earliest `menuDate` found in the `Dish` table.
    """
    return session.query(Dish.menuDate).order_by(Dish.menuDate.asc()).first()[0]


def get_written_forms(db_session: Session) -> Tuple[dict, dict, dict, dict, dict, dict]:
    """
    Retrieves written forms for additives, allergens, and meats from the directory.

    Parameters:
        session (Session): The SQLAlchemy session used to query the database.

    Returns:
        tuple: A tuple containing six dictionaries:
            - additives_dict: A dictionary mapping additives to their written forms.
            - additives_dict_eng: A dictionary mapping additives to their English written forms.
            - allergens_dict: A dictionary mapping allergens to their written forms.
            - allergens_dict_eng: A dictionary mapping allergens to their English written forms.
            - meats_dict: A dictionary mapping meats to their written forms.
            - meats_dict_eng: A dictionary mapping meats to their English written forms.
    """
    additives_dict = {}
    additives_dict_eng = {}
    allergens_dict = {}
    allergens_dict_eng = {}
    meats_dict = {}
    meats_dict_eng = {}
    
    directory_entries = db_session.query(Directory).all()
    
    for entry in directory_entries:
        if entry.additives:
            additives_dict[entry.additives] = entry.additives_written
            additives_dict_eng[entry.additives] = entry.additives_written_eng
        if entry.allergens:
            allergens_dict[entry.allergens] = entry.allergens_written
            allergens_dict_eng[entry.allergens] = entry.allergens_written_eng
        if entry.meats:
                meats_dict[entry.meats] = entry.meats_written
                meats_dict_eng[entry.meats] = entry.meats_written_eng
            
    return additives_dict, additives_dict_eng, allergens_dict, allergens_dict_eng, meats_dict, meats_dict_eng


def get_user_name(db_session: Session, username: str) -> str:
    """
    Retrieves a user record from the database based on the provided username.

    Parameters:
        db_session (Session): The SQLAlchemy session used to query the database.
        username (str): The username of the user to retrieve.

    Returns:
        User: The user object if found, otherwise None.
    """
    return db_session.query(User).filter_by(username=username).first()


def get_ratings_of_the_week(db_session: Session, week_dates: List[str]) -> List[Rating]:
    """
    Retrieves the average ratings and rating counts for dishes over a specified week.
    It calculates the average rating and the count of ratings for each dish, grouped by location, menu, and menu date.

    Parameters:
        session (Session): The SQLAlchemy session used to query the database.
        week_dates (List[str]): A list of date strings (in 'YYYY-MM-DD' format) representing the week for which to retrieve ratings.

    Returns:
        pandas.DataFrame: A DataFrame containing the following columns:
            - 'location': The location where the dish is served.
            - 'menu': The name of the dish.
            - 'menuDate': The date the dish is available.
            - 'avg_rating': The average rating of the dish, rounded to 2 decimal places.
            - 'rating_count': The number of ratings for the dish on that date.
    """
    query = db_session.query(
        func.avg(Rating.rating).label('avg_rating'),
        func.count(Rating.id).label('rating_count'),
        Dish.location,
        Dish.menu,
        DishEng.menuEng,
        Dish.menuDate
        ).join(Dish, Rating.menu_id == Dish.id
        ).join(DishEng, Dish.id == DishEng.menu_id
        ).filter(Dish.menuDate.in_(week_dates)
        ).group_by(Dish.id,Dish.menu,DishEng.menuEng,Dish.location,Dish.menuDate).all()

    result = [
        {
            'location': rating.location,
            'menu': rating.menu,
            'menuEng': rating.menuEng,
            'menuDate': rating.menuDate,
            'avg_rating': round(rating.avg_rating, 2),
            'rating_count': rating.rating_count
        } for rating in query
    ]
    return pd.DataFrame(result)

def get_top_three_dishes(session: Session, lang, week_dates: List[str]) -> List[Tuple[str, float]]:
    """
    Retrieves the top three rated dishes for a given week based on user ratings.
    The function filters dishes that have ratings in the specified week and are categorized as "Hauptspeise" (main dishes).
    It calculates the average rating and count of ratings for each dish.

    Parameters:
        session (Session): The SQLAlchemy session used to query the database.
        lang (str): The language preference for the dish names ('de' for German, other values for English).
        week_dates (List[str]): A list of date strings (in 'YYYY-MM-DD' format) representing the week to filter dishes.

    Returns:
        List[dict]: A list of dictionaries containing the top three dishes, with the following keys:
            - 'dish_name': The name of the dish (in the preferred language).
            - 'avg_rating': The average rating of the dish, rounded to 2 decimal places.
            - 'rating_count': The total number of ratings for the dish.
    """
    # get dishes for the week
    top_dishes = session.query(
        Dish.menu,
        DishEng.menuEng,
        func.avg(Rating.rating).label('avg_rating'),
        func.count(Rating.id).label('rating_count')
        ).filter(
        Dish.menuDate.in_(week_dates),
        Dish.menu != "NA",
        Course.course == "Hauptspeise"
        ).join(Rating, Dish.id == Rating.menu_id
        ).join(Course, Dish.id == Course.menu_id
        ).join(DishEng, Dish.id == DishEng.menu_id
        ).group_by(Dish.id,Dish.menu,DishEng.menuEng
        ).order_by(func.avg(Rating.rating).desc()
        ).limit(3).all()
    
    result = [
        {
            'dish_name': dish.menu if lang == 'de' else dish.menuEng,
            'avg_rating': round(dish.avg_rating, 2),
            'rating_count': dish.rating_count
        } for dish in top_dishes
    ]

    return result

def get_top_three_mensas(db_session: Session,week_dates: List[str]) -> List[Tuple[str, float]]:
    """
    Retrieves the top three rated mensas for a given week based on the average user ratings of main dishes

    Parameters:
        db_session (Session): The SQLAlchemy session used to query the database.
        week_dates (List[str]): A list of date strings (in 'YYYY-MM-DD' format) representing the week to filter dishes.

    Returns:
        List[dict]: A list of dictionaries containing the top three mensas, with the following keys:
            - 'location': The name of the mensa (location).
            - 'avg_rating': The average rating of the dishes served at that mensa, rounded to 2 decimal places.
            - 'rating_count': The total number of ratings for the dishes served at that mensa.
    """

    # get dishes for the week
    top_dishes = db_session.query(
        Dish.location,
        func.avg(Rating.rating).label('avg_rating'),
        func.count(Rating.id).label('rating_count')
        ).filter(
        Dish.menuDate.in_(week_dates),
        Dish.menu != "NA",
        Course.course == "Hauptspeise"
        ).join(Rating, Dish.id == Rating.menu_id
        ).join(Course, Dish.id == Course.menu_id
        ).group_by(Dish.location
        ).order_by(func.avg(Rating.rating).desc()
        ).limit(3).all()
    
        # Format result
    result = [
        {
            'location': dish.location,
            'avg_rating': round(dish.avg_rating, 2),
            'rating_count': dish.rating_count
        } for dish in top_dishes
    ]

    return result


def get_dishes_and_rating_by_week(db_session: Session, week_dates: List[str]):
    """
    Retrieves dishes and their ratings for a specific week.

    Parameters:
        db_session (Session): The SQLAlchemy session used to query the database.
        week_dates (List[str]): A list of date strings (in 'YYYY-MM-DD' format) representing the week to filter dishes.

    Returns:
        pandas.DataFrame: A DataFrame containing the following columns:
            - 'menu_id': The unique ID of the dish.
            - 'menu': The name of the dish.
            - 'rating': The rating of the dish.
            - 'location': The location (mensa) where the dish is served.
    """
    query = db_session.query(
        Dish.menu,
        Dish.id,
        Rating.rating,
        Dish.location
        ).join(Rating, Dish.id == Rating.menu_id
        ).filter(Dish.menuDate.in_(week_dates), Dish.menu != "NA",Rating.rating != None
        ).all()
    
    result = [
        {
            'menu_id': dish.id,
            'menu': dish.menu,
            'rating': dish.rating,
            'location': dish.location
        } for dish in query
    ]
    return pd.DataFrame(result)

def get_total_ratings_by_user(db_session: Session, username: str) -> int:
    """
    Retrieves the total number of ratings submitted by a specific user.

    Parameters:
        session (Session): The SQLAlchemy session used to query the database.
        username (str): The username of the user whose ratings are to be counted.

    Returns:
        int: The total count of ratings submitted by the user.
    """
    return db_session.query(func.count(Rating.id)).filter(Rating.user_name == username).scalar()

def get_first_rating_date_of_user(db_session: Session, username: str) -> str:
    """
    Retrieves the date of the first rating submitted by a specific user.

    Parameters:
        session (Session): The SQLAlchemy session used to query the database.
        username (str): The username of the user whose first rating date is to be fetched.

    Returns:
        str: The date of the first rating in 'YYYY-MM-DD' format, or None if no ratings exist.
    """
    first_rating = db_session.query(func.min(Rating.timestamp)).filter(Rating.user_name == username).scalar()
    if first_rating:
        return first_rating.strftime('%Y-%m-%d')
    return None

def get_dishes_of_user(db_session: Session, username: str) -> List[dict]:
    """
    Retrieves a list of dishes rated by a specific user, along with their ratings.

    Parameters:
        session (Session): The SQLAlchemy session used to query the database.
        username (str): The username of the user whose rated dishes are to be fetched.

    Returns:
        list: A list of dictionaries where each dictionary contains the following keys:
            - 'menu_id': The unique ID of the dish.
            - 'menu': The name of the dish in the default language.
            - 'menuEng': The name of the dish in English.
            - 'rating': The rating given by the user for that dish.
    """
    rated_dishes = db_session.query(
        Dish.id,
        Dish.menu,
        DishEng.menuEng,
        Rating.rating
    ).join(Rating, Dish.id == Rating.menu_id
    ).outerjoin(DishEng, Dish.id == DishEng.menu_id
    ).filter(Rating.user_name == username
    ).order_by(Rating.rating.desc()
    ).all()

    menu_list = [{"menu_id": item[0],"menu":item[1],"menuEng": item[2], "rating": item[3]} for item in rated_dishes]

    return menu_list

def get_favorite_mensas_of_user(db_session: Session, username: str, lang: str = 'en', limit: int = 3) -> List[Tuple[str, float]]:
    """
    Retrieves the top favorite locations for a specific user based on their average ratings.

    Parameters:
        session (Session): The SQLAlchemy session used to query the database.
        username (str): The username of the user whose favorite mensas are to be fetched.
        lang (str, optional): The preferred language for the mensa names ('en' for English, default is 'en').
        limit (int, optional): The maximum number of favorite mensas to retrieve (default is 3).

    Returns:
        List[Tuple[str, float]]: A list of tuples, each containing:
            - The name of the mensa (in the preferred language).
            - The average rating for that mensa (rounded to 2 decimal places).
    """

    favorite_mensas = db_session.query(
        Dish.location,
        DishEng.locationEng,
        func.avg(Rating.rating).label('avg_rating')
    ).join(Rating, Dish.id == Rating.menu_id
    ).outerjoin(DishEng, Dish.id == DishEng.menu_id
    ).filter(Rating.user_name == username
    ).group_by(Dish.location, DishEng.locationEng
    ).order_by(func.avg(Rating.rating).desc()
    ).limit(limit).all()

    return [(mensa.locationEng if lang == 'en' and mensa.locationEng else mensa.location, round(mensa.avg_rating, 2)) for mensa in favorite_mensas]

def get_dishes_history(db_session):
    """
    Retrieves the history of dishes from the database and returns it as a DataFrame.

    Parameters:
        db_session (Session): The SQLAlchemy session used to query the database.

    Returns:
        pandas.DataFrame: A DataFrame containing the dish history with the following columns:
            - All relevant columns from the `DishHistory` table, including `menuDate`.

    Notes:
        - The function queries the `DishHistory` table to fetch all historical dish records.
        - After fetching the data, it uses the `convert_dblist_to_df` function to convert the results into a pandas DataFrame.
        - The `menuDate` column is converted to a pandas `datetime` format for easier date manipulation.
    """

    dishes = db_session.query(DishHistory).all()
    df = convert_dblist_to_df(dishes)
    df['menuDate'] = pd.to_datetime(df['menuDate'],format='%Y-%m-%d')


    return df

def get_combined_dishes(db_session: Session):
    """
    Combines all rows from the DishHistory table with rows from the Dish table that are not already in the `DishHistory`.
    The result is returned as a pandas DataFrame. This is needed for price prediction and for the price development plot.

    Parameters:
        db_session (Session): The SQLAlchemy session used to query the database.

    Returns:
        pandas.DataFrame: A DataFrame containing the combined dishes with the following columns:
            - 'menu': The name of the dish.
            - 'menuDate': The date when the dish was served.
            - 'menuLine': The category or line of the dish (e.g., "Hauptspeise").
            - 'studentPrice': The price of the dish for students.
            - 'guestPrice': The price of the dish for guests.
            - 'icons_clean': Cleaned icons for the dish.
    """
    # Select all rows from DishHistory
    dish_history_query = select(DishHistory.menu, DishHistory.menuDate, DishHistory.menuLine, DishHistory.studentPrice,DishHistory.guestPrice,DishHistory.icons_clean)

    # Select rows from Dish not in DishHistory
    dish_not_in_history_query = select(
        Dish.menu, Dish.menuDate, Dish.menuLine, Dish.studentPrice,Dish.guestPrice,FiltersClean.icons_clean
    ).join(
        FiltersClean, Dish.id == FiltersClean.menu_id
    ).where(
        not_(
            db_session.query(DishHistory)
            .filter(
                and_(
                    Dish.menu == DishHistory.menu,
                    Dish.menuDate == DishHistory.menuDate,
                    Dish.menuLine == DishHistory.menuLine                   
                )
            )
            .exists()
        )
    )

    # Combine both queries
    combined_query = union(dish_history_query, dish_not_in_history_query)
    result = db_session.execute(combined_query).fetchall()

    # Convert the result to a pandas DataFrame
    df = pd.DataFrame(result, columns=["menu", "menuDate", "menuLine", "studentPrice", "guestPrice","icons_clean"])
    df['menuDate'] = pd.to_datetime(df['menuDate'],format='%Y-%m-%d')

    # in the Dish table, combine categories that belong together
    df.loc[df['menuLine'] == 'Auswahlgericht vegan 2', 'menuLine'] = 'Auswahlgericht vegan'
    df.loc[df['menuLine'] == 'Auswahlgericht veget. 2', 'menuLine'] = 'Auswahlgericht veget.'
    df.loc[df['menuLine'] == 'Auswahlgericht 2', 'menuLine'] = 'Auswahlgericht'
    df.loc[df['menuLine'] == 'mensaVital vegan 2', 'menuLine'] = 'mensaVital vegan'
    
    # remove Pizza menuLine as this this was a mistake by the mensa online page
    df = df[df['menuLine'] != "Pizza"]

    return df

def get_embedding_cluster(db_session: Session) -> pd.DataFrame:
    """
    Retrieves the embedding clusters from the database and converts them into a pandas DataFrame.

    Parameters:
        db_session (Session): The SQLAlchemy session used to query the database.

    Returns:
        pandas.DataFrame: A DataFrame containing the embedding clusters with the following columns:
            - All relevant columns from the `EmbeddingCluster` table, including the centroid.
    """
    # get cluster embeddings
    query = db_session.query(EmbeddingCluster).all()

    # transform to DataFrame
    df = convert_dblist_to_df(query)

    # convert centroid to numpy array
    df['centroid'] = df['centroid'].apply(lambda x: np.array(eval(x)) if isinstance(x, str) else np.array(x))

    return df

def get_dish_vector_for_week(db_session, week_dates):
    """
    Retrieves the dish vectors for the main dishes served during a specific week.

    Parameters:
        db_session (Session): The SQLAlchemy session used to query the database.
        week_dates (List[str]): A list of dates representing the week for which the dish vectors are to be fetched.

    Returns:
        pandas.DataFrame: A DataFrame containing the following columns:
            - 'menu_id': The unique ID of the dish.
            - 'embedding': The embedding vector associated with the dish.
    """
    # get dish_id and embedding for main dishes
    query = db_session.query(
        Dish.id,
        Embedding.embedding
        ).filter(
        Dish.menuDate.in_(week_dates),Course.course == "Hauptspeise"
        ).join(Embedding, Dish.id == Embedding.menu_id
        ).join(Course, Dish.id == Course.menu_id).all()

    # convert to DataFrame
    df = pd.DataFrame(query, columns=["menu_id", "embedding"])

    # convert embedding to numpy array
    df['embedding'] = df['embedding'].apply(lambda x: np.array(eval(x)) if isinstance(x, str) else np.array(x))
    
    return df

# map cluster names from English to German
cluster_translation_dict = {
    'Asian': 'Asiatisch',
    'Balkan': 'Balkan',
    'Central European': 'Mitteleuropäisch',
    'French': 'Französisch',
    'German': 'Deutsch',
    'Greek': 'Griechisch',
    'Italian': 'Italienisch',
    'Creole': 'Kreolisch',
    'Mediterranean': 'Mediterran',
    'Mexican': 'Mexikanisch',
    'Oriental': 'Orientalisch',
    'Thai': 'Thailändisch',
    'Bavarian': 'Bayerisch',
    'Other': 'Andere',
    'Swabian': 'Schwäbisch',
    'Eastern European': 'Osteuropäisch',
}

def get_cluster_similarity(db_session: Session, user_name: str, lang: str) -> pd.DataFrame:
    """
    Computes the cosine similarity between a user's vector and the centroids of different embedding clusters.
    The result is returned as a DataFrame with scaled similarity scores.

    Parameters:
        db_session (Session): The SQLAlchemy session used to query the database.
        user_name (str): The username of the user whose vector will be compared to the cluster centroids.
        lang (str): The language preference ('de' for German, otherwise English).

    Returns:
        pandas.DataFrame: A DataFrame containing the following columns:
            - 'centroid': The centroid of the cluster.
            - 'cosine_similarity': The cosine similarity between the user's vector and the cluster centroid.
            - 'min_max_scaler': The normalized cosine similarity using MinMax scaling.
            - 'scaled': A scaled similarity score, where values are centered around 0 (scaled between -1 and 1).
            - 'cluster_name': The name of the cluster (translated to German if lang is 'de').
    """

    # get user vector
    query = get_user_by_username(db_session,user_name)
    user_vector = np.array(eval(query.user_vector))

    # get centroid df
    centroid_df = get_embedding_cluster(db_session)

    # get centroids
    centroids = np.vstack(centroid_df['centroid'])

    # Compute cosine similarity
    similarity_scores = cosine_similarity([user_vector], centroids)[0]
    centroid_df['cosine_similarity'] = similarity_scores

    if lang == "de":
        centroid_df['cluster_name'] = centroid_df['cluster_name'].apply(lambda x: cluster_translation_dict[x])

    # scale cosine similarity
    scaler = MinMaxScaler()
    centroid_df["min_max_scaler"] = scaler.fit_transform(centroid_df.cosine_similarity.values.reshape(-1,1))

    # scale similarity: similarity of 1 is scaled to one, similarity of 0.5 is 0
    centroid_df["scaled"] = centroid_df["cosine_similarity"].apply(lambda x: (x - 0.5) / 0.5)

    return centroid_df


def get_week_recommended_dishes(db_session: Session, week_dates: str, user_name: str, selected_lang:str) -> pd.DataFrame:
    """
    Retrieve the recommended dishes for a specific week based on the user's preferences and the selected language.

    This function queries the dishes available for the given week, excluding any dishes marked as "NA".
    It joins several related tables (FiltersClean, Recipe, Description, DishEng, Course, Embedding, and PriceClean)
    to gather additional details about each dish. It then computes the cosine similarity between the user's
    preferences and the embedding for each dish, if a valid user vector is available.

    Parameters:
    - db_session: The SQLAlchemy session used to query the database.
    - week_dates: A list of dates (in 'YYYY-MM-DD' format) representing the week.
    - user_name: The username of the user for whom recommendations are being generated.
    - selected_lang: The language for the dish descriptions and other text. Can be either 'de' (German) or 'en' (English).

    Returns:
    - A Pandas DataFrame containing the recommended dishes for the week with columns:
      - Dish details: id, menu, menuLine, studentPrice, guestPrice, allergens, additives, location.
      - Translations based on the selected language: menuLineEng, menuEng, description_en, recipe_en, course_eng.
      - Additional details: icons_clean, recipe_de, description_de, embedding, course, studentPrice_imputed, guestPrice_imputed.
      - If a user vector exists, a column `cosine_similarity` will be included, representing the similarity between
        the user's vector and the dish's embedding.
    """

    # get dishes for the week
    query = db_session.query(Dish).filter(
        Dish.menuDate.in_(week_dates),
        Dish.menu != "NA"
    )

    # Join additional tables
    query = (
        query
        .outerjoin(FiltersClean, Dish.id == FiltersClean.menu_id)
        .outerjoin(Recipe, Dish.id == Recipe.menu_id)
        .outerjoin(Description, Dish.id == Description.menu_id)
        .outerjoin(DishEng, Dish.id == DishEng.menu_id)
        .outerjoin(Course, Dish.id == Course.menu_id)
        .outerjoin(Embedding, Dish.id == Embedding.menu_id)
        .outerjoin(PriceClean, Dish.id == PriceClean.menu_id)
        .add_columns(
            Dish.id,Dish.menu,Dish.menuLine,Dish.studentPrice,Dish.guestPrice,Dish.allergens,Dish.additives,Dish.location,
            FiltersClean.icons_clean.label("icons_clean"),
            Recipe.recipe_de,
            Description.description_de,
            Embedding.embedding,
            Course.course,
            PriceClean.studentPrice_imputed,
            PriceClean.guestPrice_imputed,
            *([DishEng.menuLineEng, DishEng.menuEng, Description.description_en, Recipe.recipe_en, Course.course_eng] if selected_lang == "en" else [])
        )
    )
    # Convert to DataFrame
    df = pd.DataFrame(query.all())

    # get user vector
    user_vector = get_user_vector(user_name, db_session)

    # Compute cosine similarity if user vector exists
    if user_vector:
        df["cosine_similarity"] = df["embedding"].apply(lambda x: compute_cosine_similarity(x, user_vector))

    return df

def get_weekly_recommendation_dict(db_session, user_name, lang):
    """
    Generate a weekly recommendation dictionary for the user, based on their preferences and the selected language.

    This function retrieves recommended dishes for the week and calculates a score for each dish
    based on cosine similarity between the user's vector and the dish's embedding. The result is formatted into a dictionary
    where each day contains the top dish recommended for that day and course.

    Parameters:
    - db_session: The SQLAlchemy session used to query the database.
    - user_name: The username of the user for whom recommendations are being generated.
    - lang: The language in which the dish details should be returned. Can be either 'de' (German) or 'en' (English).

    Returns:
    - A dictionary containing the top recommended dish for each day of the week
    """

    # get dishes for the week and compute user vector with user_name
    df = get_week_recommended_dishes(db_session, get_weekday_dates(), user_name, lang)

    # format prices
    df = format_price_column(df, 'studentPrice', 'studentPrice_imputed')
    df = format_price_column(df, 'guestPrice', 'guestPrice_imputed')

    # add day of week to dataframe
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    course_order = ['Hauptspeise', 'Beilage', 'Nachspeise']

    # Add day of week and course as categorical columns
    df['day_of_week'] = pd.Categorical(pd.to_datetime(df['Dish'].apply(lambda x: x.menuDate)).dt.day_name(),categories=day_order,ordered=True)

    # Add course as categorical column
    df['course'] = pd.Categorical(df['course'],categories=course_order,ordered=True)
    df.drop(columns=["Dish"], inplace=True)

    # Translate day of week to German if selected
    day_translation_dict = {
        'Monday': 'Montag',
        'Tuesday': 'Dienstag',
        'Wednesday': 'Mittwoch',
        'Thursday': 'Donnerstag',
        'Friday': 'Freitag'
    }

    if lang == "de":
        df['day_of_week'] = df['day_of_week'].apply(lambda x: day_translation_dict[x])

    # Get top dish for each day and course
    top_dishes = (df.sort_values('cosine_similarity', ascending=False).groupby(["day_of_week", "course"], group_keys=False).head(1))
    
    # Calculate cosine similarity and scale to 0-100
    top_dishes['taste_score'] = (top_dishes['cosine_similarity'] * 100).round(0).astype(int)

    # Organize the final result into a dictionary by day
    top_dishes.sort_values(["day_of_week", "course"], inplace=True)

    # Group the data by day of the week
    grouped_menu_data = {day: top_dishes[top_dishes["day_of_week"] == day].to_dict(orient="records") for day in top_dishes["day_of_week"].unique()}

    return grouped_menu_data

def get_prevornext_weekday_dates(start_date: str, direction: str) -> list:
    """
    Get the week dates (Monday to Friday) for the previous or next week.

    Args:
    - start_date (str): The date of a Monday in "YYYY-MM-DD" format.
    - direction (str): "back" for the previous week, "next" for the next week.

    Returns:
    - list: List of week dates (Monday to Friday) as strings in "YYYY-MM-DD" format.
    """
    # based on input start date
    current_monday = datetime.strptime(start_date, "%Y-%m-%d")
    
    # get the offset based on the direction. If user clicks on "back", get week dates for the previous week
    if direction == "back":
        offset = -7
    else:
        offset = 7
    
    # Calculate the Monday of the target week
    target_monday = current_monday + timedelta(days=offset)
    
    # Generate the week dates (Monday to Friday)
    week_dates = [(target_monday + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(5)]
    
    return week_dates

def get_weekday_dates() -> list:
    """
    Get the dates for the current week's weekdays (Monday to Friday).

    This function calculates the dates of the current week starting from Monday 
    and returns them as a list of strings in the format "YYYY-MM-DD". The list 
    includes dates for Monday to Friday.

    Returns:
    - A list of strings
    """

    today = datetime.today()
    start_of_week = today - timedelta(days=today.weekday())

    weekdays = [(start_of_week + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(5)]
    return weekdays


def get_current_month_spending(session: Session, user_name:str, lang='en') -> pd.DataFrame:
    """
    Calculate the total spending for the current month by the user for different categories and return it as a DataFrame.

    Parameters:
    - session (Session): The SQLAlchemy session object for database interaction.
    - user_name (str): The username of the user for whom the spending is calculated.
    - lang (str): The language code for the month name ('en' or 'de'). Default is 'en'.

    Returns:
    - pd.DataFrame: A DataFrame containing the total spending for the current month
    """

    current_year = datetime.now().year
    current_month = datetime.now().month
    current_month_name = get_month_name(current_month, lang)

    result = session.query(
        func.sum(Dish.studentPrice).label('total_student_spending'),
        func.sum(Dish.guestPrice).label('total_guest_spending'),
        func.sum(Dish.pupilPrice).label('total_pupil_spending'),
    ).select_from(Rating).join(
        Dish, Rating.menu_id == Dish.id
    ).filter(
        Rating.user_name == user_name,
        Rating.on_rating_page == False,
        extract('year', Rating.timestamp) == current_year,
        extract('month', Rating.timestamp) == current_month,
        Dish.studentPrice > 0
    ).one_or_none()
        
    df = pd.DataFrame([{
        'year': current_year,
        'month': current_month,
        'month_name': current_month_name,
        'student': result.total_student_spending if result and result.total_student_spending else 0.0,
        'guest': result.total_guest_spending if result and result.total_guest_spending else 0.0,
        'pupil': result.total_pupil_spending if result and result.total_pupil_spending else 0.0,
    }])  
        
    return df



def get_past_6_month_spending(session: Session, user_name: str, lang='en') -> pd.DataFrame:
    """
    Calculate the total spending for the past 6 months by the user and return it as a DataFrame.

    Parameters:
    - session (Session): The SQLAlchemy session object for database interaction.
    - user_name (str): The username of the user for whom the spending is calculated.
    - lang (str): The language code for the month name ('en' or 'de'). Default is 'en'.

    Returns:
    - pd.DataFrame: A DataFrame containing the total spending for each of the past 6 months
    """

    current_year = datetime.now().year
    current_month = datetime.now().month

    # List to store data
    month_data = []

    for i in [0, 1, 2, 3, 4, 5]:
        # Calculate the month and year for each of the last 6 months
        month_offset = (current_month - i) if (current_month - i) > 0 else (12 + (current_month - i))
        year_offset = current_year if current_month - i > 0 else current_year - 1
        month_name = get_month_name(month_offset, lang)

        result = session.query(
            func.sum(Dish.studentPrice).label('total_student_spending'),
            func.sum(Dish.guestPrice).label('total_guest_spending'),
            func.sum(Dish.pupilPrice).label('total_pupil_spending')
        ).select_from(Rating).join(
            Dish, Rating.menu_id == Dish.id
        ).filter(
            Rating.user_name == user_name,
            Rating.on_rating_page == False,
            extract('year', Rating.timestamp) == year_offset,
            extract('month', Rating.timestamp) == month_offset
        ).one_or_none()

        # Append data for current month
        month_data.append({
            'year': year_offset,
            'month': month_offset,
            'month_name': month_name,
            'student': result.total_student_spending if result and result.total_student_spending else 0.0,
            'guest': result.total_guest_spending if result and result.total_guest_spending else 0.0,
            'pupil': result.total_pupil_spending if result and result.total_pupil_spending else 0.0
        })

    return pd.DataFrame(month_data)

    
def get_cluster_similarity_for_week(db_session: Session, week_dates: List[str], lang: str) -> pd.DataFrame:
    """
    Compute the cluster similarity for dishes served during the selected week and return a DataFrame
    with the count of dishes for each cluster.

    Parameters:
    - db_session (Session): The SQLAlchemy session object for interacting with the database.
    - week_dates (list): List of date strings for the week (['2025-01-26', '2025-01-27', ...]).
    - lang (str): The language code for cluster names

    Returns:
    - pd.DataFrame: A DataFrame containing the cluster names and the number of dishes assigned to each cluster.
    """

    # Get centroid df
    centroid_df = get_embedding_cluster(db_session)

    # remove German and Other clusters because we stick to schwäbisch and bayrisch which are part of the German cluster
    if lang == 'de':
        centroid_df['cluster_name'] = centroid_df['cluster_name'].apply(lambda x: cluster_translation_dict[x])
        centroid_df = centroid_df[~centroid_df['cluster_name'].isin(['Deutsch','Andere'])]
    else:
        centroid_df = centroid_df[~centroid_df['cluster_name'].isin(['German','Other'])]

    # Get centroids
    centroids = np.vstack(centroid_df['centroid'])

    # Get embedding matrices for the selected dates
    df = get_dish_vector_for_week(db_session, week_dates)

    # Initialize a list to store results
    cluster_names = []
    max_similarities = []

    # Iterate over each row in df and compute cosine similarity
    for _, row in df.iterrows():
        # Get the embedding for the current row
        dish_vector = row['embedding']

        # Compute cosine similarity
        similarity_scores = cosine_similarity([dish_vector], centroids)[0]
        centroid_df['cosine_similarity'] = similarity_scores

        # Get the cluster_name with the highest similarity
        max_idx = centroid_df['cosine_similarity'].idxmax()
        max_similarity = centroid_df.loc[max_idx, 'cosine_similarity']

        # If the max similarity is less than 0.75, assign 'other' as the cluster. So only classify into category if similarity is larger than 0.75
        if max_similarity < 0.75:
            cluster_name = 'Other' if lang == 'en' else 'Andere'
        else:
            cluster_name = centroid_df.loc[max_idx, 'cluster_name']

        cluster_names.append(cluster_name)
        max_similarities.append(max_similarity)

    # Add the cluster names to the DataFrame
    df['cluster_name'] = cluster_names
    df['max_similarity'] = max_similarities

    # get number of meals per cluster
    cluster_counts = df.groupby('cluster_name').size().reset_index(name='count')

    return cluster_counts