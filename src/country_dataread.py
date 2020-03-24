import pandas
import json
import numpy as np

def readcorona(location: str):
	df = pandas.read_csv(location)
	df = df[['Country,Other','TotalCases','TotalDeaths','Tot Cases/1M pop']]
	df = df[df['TotalDeaths'] > 0] 
	df['Pop'] = df['TotalCases'] * 1 / df['Tot Cases/1M pop']
	df = df[['Country,Other', 'TotalDeaths', 'Pop']]
	df.rename(columns={'Country,Other': 'Name'}, inplace = "JATAK")
	df = df[:-1]
	df['TotalDeaths'] /= df['Pop']
	return df

def readdatasets(location: str):
	dfs = []
	for dataset in ['age', 'continent', 'density', 'gdp','healthcare', 'tourists']:
		dfs.append( pandas.read_csv(f"{location}/{dataset}.csv") )
	return dfs

def find_countries(corona, dfs, verbose=False):
	for df in dfs: corona[ df.columns[-1] ] = np.nan
	for i, country in corona.iterrows():
		for df in dfs:
			found = False
			for j, data_country in df.iterrows():
				if country["Name"] in data_country["Name"]:
					if found and verbose:
						print(f'{df.columns[-1]} already found for:', country["Name"], data_country["Name"])
					else:
						corona[df.columns[-1]][i] = data_country[df.columns[-1]]
						found = True
			if not found and verbose:
				print(f'{df.columns[-1]} not found for:', country["Name"])
	return corona

def cleanup(corona):
	corona = corona.dropna()
	corona["Tourists"] /= corona["Pop"]
	country_names = list(corona["Name"])
	deaths = np.array(corona["TotalDeaths"])

	corona = corona[["Age","Density", "GDP", "Healthcare", "Tourists", "Continent"]]	
	feature_names = [str(i) for i in corona.columns]
	
	uniques = np.unique(corona["Continent"])
	corona = corona.replace(uniques, range(len(uniques)))
	return corona.to_numpy(), deaths, country_names, feature_names

def get_data():
	corona = readcorona("saved_data/corona_latest.csv")
	dfs = readdatasets("saved_data")
	corona = find_countries(corona, dfs)
	return cleanup(corona)

if __name__ == "__main__":
	X, y, country_names, feature_names = get_data() 
	np.savez('saved_data/corona.npz', X=X, y=y)
	with open('saved_data/names.json', 'w+') as f:
		json.dump({'countries': country_names, 'features': feature_names}, f, indent=4) 
