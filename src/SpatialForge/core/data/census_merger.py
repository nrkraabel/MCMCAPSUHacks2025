import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point, shape
from shapely.ops import unary_union
import os
import requests
import zipfile
from tqdm import tqdm
import sys


def download_census_blocks(state_fips='42'):
    """
    Download census block data for Pennsylvania (FIPS code 42)
    
    Returns:
    - Path to downloaded shapefile
    """
    # Create directory for data
    os.makedirs('census_data', exist_ok=True)
    
    # Pennsylvania FIPS code is 42
    url = f"https://www2.census.gov/geo/tiger/TIGER2020/TABBLOCK20/tl_2020_{state_fips}_tabblock20.zip"
    output_zip = "census_data/pa_blocks.zip"
    output_dir = "census_data/pa_blocks"
    
    # Check if already downloaded
    if os.path.exists(output_dir):
        print(f"Census block data already exists in {output_dir}")
        # Find shapefile in directory
        for file in os.listdir(output_dir):
            if file.endswith(".shp"):
                return os.path.join(output_dir, file)
        raise FileNotFoundError(f"No shapefile found in {output_dir}")
    
    # Download the data
    print(f"Downloading census block data from {url}...")
    response = requests.get(url, stream=True)
    
    if response.status_code != 200:
        raise Exception(f"Failed to download data: {response.status_code}")
    
    # Save the zip file
    with open(output_zip, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    # Extract the data
    print(f"Extracting zip file to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    with zipfile.ZipFile(output_zip, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
    
    # Find the shapefile
    for file in os.listdir(output_dir):
        if file.endswith(".shp"):
            return os.path.join(output_dir, file)
    
    raise FileNotFoundError(f"No shapefile found in {output_dir}")

def download_census_pl_data(state_fips='42'):
    """
    Download PL 94-171 data for Pennsylvania
    
    Returns:
    - Path to downloaded data file
    """
    # Create directory for data
    os.makedirs('census_data', exist_ok=True)
    
    # URL for Pennsylvania PL-94-171 data
    # This is the 2020 Census Redistricting Data
    url = f"https://www2.census.gov/programs-surveys/decennial/2020/data/01-Redistricting_File--PL_94-171/{state_fips}-Pennsylvania/pa2020.pl.zip"
    output_zip = "census_data/pa_pl94_171.zip"
    output_dir = "census_data/pa_pl94_171"
    
    # Check if already downloaded
    if os.path.exists(output_dir):
        print(f"PL 94-171 data already exists in {output_dir}")
        return output_dir
    
    # Download the data
    print(f"Downloading PL 94-171 data from {url}...")
    response = requests.get(url, stream=True)
    
    if response.status_code != 200:
        raise Exception(f"Failed to download data: {response.status_code}")
    
    # Save the zip file
    with open(output_zip, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    # Extract the data
    print(f"Extracting zip file to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    with zipfile.ZipFile(output_zip, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
    
    return output_dir

def process_census_pl_data(pl_data_dir):
    """
    Process PL 94-171 data to extract population counts at block level
    
    Returns:
    - DataFrame with block GEOIDs and population counts
    """
    # Find the segment 1 file (contains population data)
    segment1_files = [f for f in os.listdir(pl_data_dir) if f.endswith('000012020.pl')]
    
    if not segment1_files:
        raise FileNotFoundError(f"No segment 1 file found in {pl_data_dir}")
    
    segment1_file = os.path.join(pl_data_dir, segment1_files[0])
    
    # Find the geo header file
    geo_files = [f for f in os.listdir(pl_data_dir) if f.endswith('2020.pl') and 'geo' in f.lower()]
    
    if not geo_files:
        raise FileNotFoundError(f"No geo header file found in {pl_data_dir}")
    
    geo_file = os.path.join(pl_data_dir, geo_files[0])
    
    # Read the geo header file to get block GEOIDs
    # Documentation: https://www2.census.gov/programs-surveys/decennial/2020/technical-documentation/complete-tech-docs/summary-file/2020Census_PL94_171Redistricting_StatesTechDoc.pdf
    print("Reading geographic header file...")
    geo_cols = {
        'FILEID': (0, 6),
        'STUSAB': (6, 8),
        'SUMLEV': (8, 11),
        'GEOCOMP': (11, 13),
        'GEOID': (178, 218),  # Combined geographic identifier
        'BLOCK': (226, 232)   # Block number
    }
    
    geo_df = pd.DataFrame()
    with open(geo_file, 'r') as f:
        rows = []
        for line in f:
            row = {}
            for col, (start, end) in geo_cols.items():
                row[col] = line[start:end].strip()
            rows.append(row)
        geo_df = pd.DataFrame(rows)
    
    # Filter for block level records
    block_geo_df = geo_df[geo_df['SUMLEV'] == '750']
    
    # Read the segment 1 file to get population counts
    print("Reading population data file...")
    # P1 table contains population data
    # First field is usually total population
    pop_df = pd.read_csv(segment1_file, header=None, sep='|')
    
    # Combine the data
    block_geo_df['LOGRECNO'] = block_geo_df.index.astype(str)
    pop_df['LOGRECNO'] = pop_df.index.astype(str)
    
    # Join the two dataframes
    combined_df = pd.merge(block_geo_df, pop_df, on='LOGRECNO')
    
    # Extract the population column (usually column 4)
    combined_df['POPULATION'] = combined_df.iloc[:, 4].astype(int)
    
    # Create a clean dataframe with just the GEOID and population
    result_df = combined_df[['GEOID', 'POPULATION']]
    
    return result_df

def get_census_population(election_gdf):
    """
    Get population data from census and merge with election data
    
    Parameters:
    - election_gdf: GeoDataFrame with election data
    
    Returns:
    - GeoDataFrame with election data and census population
    """
    try:
        print("Getting population data from census...")
        
        # Download census block data
        census_block_path = download_census_blocks()
        
        # Load the census block data
        print(f"Loading census block data from {census_block_path}...")
        census_gdf = gpd.read_file(census_block_path)
        
        # Try to find population in census_gdf
        if 'POP20' in census_gdf.columns:
            # Some versions of the TIGER/Line files have population directly
            print("Found population data in census blocks shapefile")
            census_gdf.rename(columns={'POP20': 'CENSUS_POP'}, inplace=True)
        else:
            # Need to download the PL 94-171 data
            print("Population data not found in shapefile, downloading PL 94-171 data...")
            pl_data_dir = download_census_pl_data()
            population_df = process_census_pl_data(pl_data_dir)
            
            # Merge population with census blocks
            print("Merging population data with census blocks...")
            census_gdf = census_gdf.merge(population_df, left_on='GEOID20', right_on='GEOID', how='left')
            census_gdf.rename(columns={'POPULATION': 'CENSUS_POP'}, inplace=True)
        
        # Check if we have population data
        if 'CENSUS_POP' not in census_gdf.columns:
            print("Census population data not available")
            return election_gdf
        
        # Convert to same CRS if needed
        if census_gdf.crs != election_gdf.crs:
            print(f"Converting census data from {census_gdf.crs} to {election_gdf.crs}")
            census_gdf = census_gdf.to_crs(election_gdf.crs)
        
        print("Performing spatial join to allocate census population to precincts...")
        # We need to aggregate census blocks to precincts
        # First, calculate centroids of census blocks
        census_gdf['geometry_centroid'] = census_gdf.geometry.centroid
        
        # Convert to GeoDataFrame with centroid geometry
        centroids_gdf = gpd.GeoDataFrame(census_gdf, geometry='geometry_centroid')
        
        # Spatial join using centroids
        joined = gpd.sjoin(centroids_gdf, election_gdf, how='left', predicate='within')
        
        # Aggregate population by precinct
        population_by_precinct = joined.groupby('index_right')['CENSUS_POP'].sum().reset_index()
        population_by_precinct.rename(columns={'index_right': 'precinct_id', 'CENSUS_POP': 'CENSUS_POP'}, inplace=True)
        
        # Merge population back to election data
        election_gdf['precinct_id'] = election_gdf.index
        election_gdf = election_gdf.merge(population_by_precinct, on='precinct_id', how='left')
        
        # Fill NaN values
        election_gdf['CENSUS_POP'] = election_gdf['CENSUS_POP'].fillna(0).astype(int)
        
        # Calculate total census population
        total_census_pop = election_gdf['CENSUS_POP'].sum()
        print(f"Total census population: {total_census_pop:,}")
        
        # Create an estimate column that uses census when available, otherwise use vote-based estimation
        election_gdf['POPULATION'] = np.where(
            election_gdf['CENSUS_POP'] > 0,
            election_gdf['CENSUS_POP'],
            (election_gdf['TOTAL_VOTES'] / 0.6).round()
        )
        
        # Calculate correlation between census and vote-based population
        valid_rows = election_gdf[(election_gdf['CENSUS_POP'] > 0) & (election_gdf['TOTAL_VOTES'] > 0)]
        if len(valid_rows) > 0:
            vote_based_pop = (valid_rows['TOTAL_VOTES'] / 0.6).round()
            correlation = np.corrcoef(valid_rows['CENSUS_POP'], vote_based_pop)[0, 1]
            print(f"Correlation between census population and vote-based estimate: {correlation:.4f}")
        
        return election_gdf
        
    except Exception as e:
        print(f"Error processing census data: {e}")
        print("Falling back to vote-based population estimate")
        return election_gdf

def load_and_prepare_election_data(shapefile_path, use_census_population=True):
    """
    Load election data and prepare it for processing.
    
    Parameters:
    - shapefile_path: Path to the election data shapefile
    - use_census_population: Whether to try to get population from census data
    
    Returns:
    - GeoDataFrame with election data and standardized column names
    """
    print(f"Loading election data from {shapefile_path}...")
    election_gdf = gpd.read_file(shapefile_path)
    
    # Map column names to standardized names
    # For presidential votes
    dem_col = next((col for col in election_gdf.columns if 'BID' in col), None)
    rep_col = next((col for col in election_gdf.columns if 'TRU' in col), None)
    other_col = next((col for col in election_gdf.columns if 'JOR' in col), None)
    
    if not dem_col or not rep_col:
        # Try alternative naming patterns
        dem_col = next((col for col in election_gdf.columns if 'PRESD' in col or 'DEM' in col), None)
        rep_col = next((col for col in election_gdf.columns if 'PRESR' in col or 'REP' in col), None)
    
    if not dem_col or not rep_col:
        print("WARNING: Could not automatically identify Democratic and Republican vote columns")
        print("Available columns:", election_gdf.columns.tolist())
        dem_col = input("Enter the column name for Democratic votes: ")
        rep_col = input("Enter the column name for Republican votes: ")
    
    print(f"Using columns: Democrat='{dem_col}', Republican='{rep_col}'")
    
    # Rename columns for consistency
    column_map = {}
    if dem_col:
        column_map[dem_col] = 'DEM_VOTES'
    if rep_col:
        column_map[rep_col] = 'REP_VOTES'
    if other_col:
        column_map[other_col] = 'OTHER_VOTES'
    
    # Apply column renaming if any mappings exist
    if column_map:
        election_gdf = election_gdf.rename(columns=column_map)
    
    # For any votes columns that weren't found and renamed, add zeros
    if 'DEM_VOTES' not in election_gdf.columns:
        election_gdf['DEM_VOTES'] = 0
    if 'REP_VOTES' not in election_gdf.columns:
        election_gdf['REP_VOTES'] = 0
    if 'OTHER_VOTES' not in election_gdf.columns:
        election_gdf['OTHER_VOTES'] = 0
    
    # Fill NaN values with zeros
    for col in ['DEM_VOTES', 'REP_VOTES', 'OTHER_VOTES']:
        election_gdf[col] = election_gdf[col].fillna(0)
    
    # Calculate total votes
    election_gdf['TOTAL_VOTES'] = election_gdf['DEM_VOTES'] + election_gdf['REP_VOTES'] + election_gdf['OTHER_VOTES']
    
    # Get population data - either from census or estimated from votes
    if use_census_population:
        election_gdf = get_census_population(election_gdf)
    else:
        # Estimate population using a turnout factor (typically ~60% turnout)
        election_gdf['POPULATION'] = (election_gdf['TOTAL_VOTES'] / 0.6).round().astype(int)
        print("Using vote-based population estimate")
    
    # Keep track of the original area for each precinct
    election_gdf['AREA'] = election_gdf.geometry.area
    
    # Check for invalid geometries
    invalid_count = election_gdf.geometry.isna().sum() + sum(1 for g in election_gdf.geometry if g is not None and not g.is_valid)
    if invalid_count > 0:
        print(f"WARNING: Found {invalid_count} invalid geometries. Attempting to fix...")
        # Try to fix invalid geometries
        election_gdf.geometry = election_gdf.geometry.buffer(0)
    
    return election_gdf

def create_state_outline(gdf):
    """
    Create a state outline from the precinct geometries
    
    Parameters:
    - gdf: GeoDataFrame with precinct geometries
    
    Returns:
    - A Shapely geometry representing the state outline
    """
    # Dissolve all precinct boundaries to get the state outline
    state_outline = unary_union(gdf.geometry)
    return state_outline

def generate_grid(state_outline, resolution=100):
    """
    Generate a grid of cells that covers the state
    
    Parameters:
    - state_outline: Shapely geometry of the state outline
    - resolution: Width of the grid in cells
    
    Returns:
    - dict containing grid parameters and cells that intersect the state
    """
    # Get the bounds of the state
    minx, miny, maxx, maxy = state_outline.bounds
    
    # Calculate the dimensions of the grid
    width = resolution
    height = int(resolution * (maxy - miny) / (maxx - minx))
    
    # Calculate cell dimensions
    cell_width = (maxx - minx) / width
    cell_height = (maxy - miny) / height
    
    print(f"Creating grid with dimensions {width}x{height} cells")
    print(f"Each cell represents approximately {cell_width:.2f} x {cell_height:.2f} units")
    
    # Create a dictionary to store the grid cells that intersect the state
    grid_cells = {}
    
    # Generate cells and check which ones intersect the state
    for i in range(height):
        for j in range(width):
            # Create cell polygon
            cell_polygon = Polygon([
                (minx + j * cell_width, miny + i * cell_height),
                (minx + (j + 1) * cell_width, miny + i * cell_height),
                (minx + (j + 1) * cell_width, miny + (i + 1) * cell_height),
                (minx + j * cell_width, miny + (i + 1) * cell_height)
            ])
            
            # Check if the cell intersects the state
            if cell_polygon.intersects(state_outline):
                # If it does, store it in our grid cells dictionary with a tuple key (i, j)
                grid_cells[(i, j)] = {
                    'geometry': cell_polygon,
                    'population': 0,
                    'dem_votes': 0,
                    'rep_votes': 0,
                    'other_votes': 0,
                    'area': cell_polygon.area
                }
    
    grid_info = {
        'width': width,
        'height': height,
        'cell_width': cell_width,
        'cell_height': cell_height,
        'minx': minx,
        'miny': miny,
        'maxx': maxx,
        'maxy': maxy,
        'cells': grid_cells
    }
    
    return grid_info

def allocate_precinct_data_to_grid(election_gdf, grid_info):
    """
    Allocate precinct data to the grid cells
    
    Parameters:
    - election_gdf: GeoDataFrame with election data
    - grid_info: Dictionary with grid parameters and cells
    
    Returns:
    - Updated grid_info with allocated data
    """
    print("Allocating precinct data to grid cells...")
    
    # Get grid parameters
    cells = grid_info['cells']
    minx = grid_info['minx']
    miny = grid_info['miny']
    cell_width = grid_info['cell_width']
    cell_height = grid_info['cell_height']
    width = grid_info['width']
    height = grid_info['height']
    
    # Process each precinct
    for idx, precinct in tqdm(election_gdf.iterrows(), total=len(election_gdf), desc="Processing precincts"):
        # Get the precinct geometry and data
        precinct_geom = precinct.geometry
        
        # Skip invalid geometries
        if precinct_geom is None or precinct_geom.is_empty:
            continue
        
        # Get the precinct data
        precinct_pop = precinct['POPULATION']
        precinct_dem = precinct['DEM_VOTES']
        precinct_rep = precinct['REP_VOTES']
        precinct_other = precinct['OTHER_VOTES']
        precinct_area = precinct['AREA']
        
        # Get the precinct bounds
        p_minx, p_miny, p_maxx, p_maxy = precinct_geom.bounds
        
        # Calculate grid cell ranges this precinct might intersect
        min_i = max(0, int((p_miny - miny) / cell_height))
        max_i = min(height - 1, int((p_maxy - miny) / cell_height) + 1)
        min_j = max(0, int((p_minx - minx) / cell_width))
        max_j = min(width - 1, int((p_maxx - minx) / cell_width) + 1)
        
        # Check each potential cell for intersection
        cell_area_sum = 0  # To track total area of intersection
        intersections = []
        
        # First pass: calculate intersections and total intersection area
        for i in range(min_i, max_i):
            for j in range(min_j, max_j):
                cell_key = (i, j)
                if cell_key in cells:
                    cell_polygon = cells[cell_key]['geometry']
                    
                    if precinct_geom.intersects(cell_polygon):
                        intersection = precinct_geom.intersection(cell_polygon)
                        intersection_area = intersection.area
                        
                        if intersection_area > 0:
                            intersections.append((cell_key, intersection_area))
                            cell_area_sum += intersection_area
        
        # Second pass: allocate data proportionally based on intersection area
        if cell_area_sum > 0:
            for cell_key, intersection_area in intersections:
                # Calculate the proportion of the precinct in this cell
                proportion = intersection_area / cell_area_sum
                
                # Allocate population and votes proportionally to this cell
                cells[cell_key]['population'] += precinct_pop * proportion
                cells[cell_key]['dem_votes'] += precinct_dem * proportion
                cells[cell_key]['rep_votes'] += precinct_rep * proportion
                cells[cell_key]['other_votes'] += precinct_other * proportion
    
    # Round values to integers
    for cell_key in cells:
        cells[cell_key]['population'] = round(cells[cell_key]['population'])
        cells[cell_key]['dem_votes'] = round(cells[cell_key]['dem_votes'])
        cells[cell_key]['rep_votes'] = round(cells[cell_key]['rep_votes'])
        cells[cell_key]['other_votes'] = round(cells[cell_key]['other_votes'])
    
    return grid_info

def create_simulator_array(grid_info, output_file="pa_election_data.npy"):
    """
    Convert the grid data to the format expected by the simulator
    
    Parameters:
    - grid_info: Dictionary with grid parameters and cell data
    - output_file: Path to save the numpy array
    
    Returns:
    - state_map: A numpy array with shape (height, width, 3) where each pixel has
                [population, red_votes, blue_votes]
    """
    print("Creating simulator array...")
    
    # Get grid dimensions
    height = grid_info['height']
    width = grid_info['width']
    cells = grid_info['cells']
    
    # Create the state map
    state_map = np.zeros((height, width, 3))
    
    # Create a mask for cells that are part of the state
    state_mask = np.zeros((height, width), dtype=bool)
    
    # Fill the state map with data from the grid cells
    for (i, j), cell_data in cells.items():
        state_map[i, j, 0] = cell_data['population']  # Population
        state_map[i, j, 1] = cell_data['rep_votes']   # Republican votes
        state_map[i, j, 2] = cell_data['dem_votes']   # Democratic votes
        
        # Mark this cell as part of the state
        state_mask[i, j] = True
    
    # Apply the state mask - set all cells outside the state to NaN or 0
    # For the simulator, we'll use 0 so it knows these areas aren't part of the state
    for i in range(height):
        for j in range(width):
            if not state_mask[i, j]:
                state_map[i, j, :] = 0
    
    # Save the state map
    np.save(output_file, state_map)
    print(f"Saved state map to {output_file}")
    
    # Print statistics
    total_pop = np.sum(state_map[:,:,0])
    total_rep = np.sum(state_map[:,:,1])
    total_dem = np.sum(state_map[:,:,2])
    
    print(f"Total population: {total_pop:,.0f}")
    print(f"Total Republican votes: {total_rep:,.0f}")
    print(f"Total Democratic votes: {total_dem:,.0f}")
    
    if total_rep + total_dem > 0:
        rep_pct = total_rep / (total_rep + total_dem) * 100
        print(f"Overall vote share: Republican {rep_pct:.1f}%, Democratic {100-rep_pct:.1f}%")
    
    return state_map, state_mask

def visualize_processed_data(state_map, state_mask, grid_info, title="Pennsylvania Election Data"):
    """
    Create visualizations of the processed data
    
    Parameters:
    - state_map: Numpy array with the processed data
    - state_mask: Boolean mask indicating which cells are part of the state
    - grid_info: Dictionary with grid parameters
    - title: Title for the visualization
    """
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle(title, fontsize=16)
    
    # Create masked versions of the data for better visualization
    masked_pop = np.ma.array(state_map[:,:,0], mask=~state_mask)
    masked_rep = np.ma.array(state_map[:,:,1], mask=~state_mask)
    masked_dem = np.ma.array(state_map[:,:,2], mask=~state_mask)
    
    # Population density
    im0 = axes[0, 0].imshow(masked_pop, cmap='viridis')
    axes[0, 0].set_title('Population Density')
    fig.colorbar(im0, ax=axes[0, 0])
    
    # Republican votes
    im1 = axes[0, 1].imshow(masked_rep, cmap='Reds')
    axes[0, 1].set_title('Republican Votes')
    fig.colorbar(im1, ax=axes[0, 1])
    
    # Democratic votes
    im2 = axes[1, 0].imshow(masked_dem, cmap='Blues')
    axes[1, 0].set_title('Democratic Votes')
    fig.colorbar(im2, ax=axes[1, 0])
    
    # Vote margin (Republican - Democratic)
    vote_margin = np.zeros_like(masked_pop)
    total_votes = masked_rep + masked_dem
    
    # Calculate margin where there are votes
    vote_margin = np.where(
        total_votes > 0,
        (masked_rep - masked_dem) / total_votes,
        0
    )
    
    # Apply the state mask
    vote_margin = np.ma.array(vote_margin, mask=~state_mask)
    
    # Create a custom diverging colormap
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        'red_white_blue', 
        [(0.8, 0, 0),      # Strong Republican (dark red)
         (1, 0.6, 0.6),    # Lean Republican (light red)
         (1, 1, 1),        # Neutral (white)
         (0.6, 0.6, 1),    # Lean Democrat (light blue)
         (0, 0, 0.8)]      # Strong Democrat (dark blue)
    )
    
    im3 = axes[1, 1].imshow(vote_margin, cmap=cmap, vmin=-1, vmax=1)
    axes[1, 1].set_title('Vote Margin (Rep - Dem) / Total Votes')
    fig.colorbar(im3, ax=axes[1, 1])
    
    # Remove axes ticks for cleaner look
    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)  # Adjust for the suptitle
    plt.savefig("pa_election_visualization.png")
    plt.show()
    
    # Create a separate visualization of just the state outline
    plt.figure(figsize=(10, 8))
    plt.imshow(state_mask, cmap='gray')
    plt.title('Pennsylvania State Mask')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("pa_state_mask.png")
    plt.show()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Process Pennsylvania election data for gerrymandering simulation")
    parser.add_argument("--shapefile", type=str ,default=r"C:\Users\nrk5343\Downloads\pa_2020\pa_2020.shp", help="Path to election data shapefile")
    parser.add_argument("--resolution", type=int, default=300, help="Resolution of the output grid (width)")
    parser.add_argument("--output", type=str, default="pa_election_data.npy", help="Output file for the numpy array")
    parser.add_argument("--census", action="store_true", help="Use census population data instead of vote-based estimates")
    parser.add_argument("--votes", action="store_true", help="Use vote-based population estimates instead of census data")
    
    args = parser.parse_args()
    
    # Determine which population method to use
    use_census = True  # Default is to try census first, fall back to votes
    if args.votes:
        use_census = False
    elif args.census:
        use_census = True
    
    # Load and prepare the election data
    election_gdf = load_and_prepare_election_data(args.shapefile, use_census_population=use_census)
    
    # Create the state outline
    state_outline = create_state_outline(election_gdf)
    
    # Generate the grid
    grid_info = generate_grid(state_outline, args.resolution)
    
    # Allocate precinct data to the grid
    grid_info = allocate_precinct_data_to_grid(election_gdf, grid_info)
    
    # Create the simulator array
    state_map, state_mask = create_simulator_array(grid_info, args.output)
    
    # Visualize the data
    visualize_processed_data(state_map, state_mask, grid_info)
    
    print("Processing complete!")

if __name__ == "__main__":
    main()