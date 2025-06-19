def get_ingredients_for_product(engine, store_product_id):
    query = f"""
        SELECT p.id AS product_id, p.name AS product_name, 
               i.id AS ingredient_id, i.name AS ingredient_name, 
               pli.qty AS qty_per_product_unit
        FROM pricelookup_store pls
        JOIN pricelookup p ON pls.pricelookup_id = p.id
        JOIN pricelookup_ingredient pli ON p.id = pli.pricelookup_id
        JOIN ingredients i ON pli.ingredient_id = i.id
        WHERE pls.id = {store_product_id};
    """
    return pd.read_sql(query, engine)


def get_store_stock_for_product_ingredients(engine, store_product_id, store_id):
    query = f"""
        SELECT ingr.id AS ingredient_id, ingr.name AS ingredient_name,
               pli.qty AS qty_per_product_unit,
               istd.stockqty, istd.lastweek_left_stockqty,
               uom.conversion_factor
        FROM pricelookup_store pls
        JOIN pricelookup p ON pls.pricelookup_id = p.id
        JOIN pricelookup_ingredient pli ON p.id = pli.pricelookup_id
        JOIN ingredients ingr ON pli.ingredient_id = ingr.id
        LEFT JOIN ingredient_store istd ON istd.ingredient_id = ingr.id AND istd.store_id = {store_id}
        LEFT JOIN unit_of_measures uom ON CASE WHEN ingr.display_uom_id != 0 THEN ingr.display_uom_id ELSE ingr.master_standard_uom END = uom.id
        WHERE pls.id = {store_product_id};
    """
    return pd.read_sql(query, engine)
