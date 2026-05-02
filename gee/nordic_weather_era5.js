// ===============================
// STEP 1: Define my study period
// ===============================

var startDate = '2010-01-01';
var endDate   = '2025-12-31';

// ===============================
// STEP 2: Load Nordic country boundaries
// ===============================

var countries = ee.FeatureCollection('USDOS/LSIB_SIMPLE/2017');

var nordics = countries.filter(
  ee.Filter.inList('country_na', ['Norway', 'Sweden', 'Finland'])
);

Map.centerObject(nordics, 4);
Map.addLayer(nordics, {color: 'red'}, 'Norway, Sweden, Finland');

print('Nordic country boundaries:', nordics);

// ===============================
// STEP 3: Load ERA5-Land temperature
// ===============================

var era5 = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR')
  .filterDate(startDate, endDate)
  .select('temperature_2m');

var tempC = era5.map(function(img) {
  return img
    .subtract(273.15)
    .rename('temp_c')
    .copyProperties(img, ['system:time_start']);
});

// I want one day to inspect on the map
var sampleTemp = tempC.first();

Map.addLayer(
  sampleTemp.clip(nordics),
  {
    min: -30,
    max: 25,
    palette: ['blue', 'cyan', 'yellow', 'orange', 'red']
  },
  'ERA5-Land temperature sample, °C'
);

print('Sample ERA5 image:', sampleTemp);

// ===============================
// STEP 4: Load WorldPop population raster
// ===============================

var pop = ee.ImageCollection('WorldPop/GP/100m/pop')
  .filterDate('2020-01-01', '2020-12-31')
  .mosaic()
  .rename('pop');

// Mask unpopulated cells
pop = pop.updateMask(pop.gt(0));

Map.addLayer(
  pop.clip(nordics),
  {
    min: 0,
    max: 1000,
    palette: ['white', 'yellow', 'orange', 'red', 'purple']
  },
  'WorldPop population raster'
);

print('Population raster:', pop);

// ===============================
// STEP 5: Test population-weighted temperature
// ===============================

var norway = nordics.filter(ee.Filter.eq('country_na', 'Norway')).first();
var norwayGeom = norway.geometry();

var testImage = tempC.filterDate('2020-01-01', '2020-01-02').first();

var weightedTemp = testImage.multiply(pop).rename('temp_x_pop');

var sums = weightedTemp.addBands(pop).reduceRegion({
  reducer: ee.Reducer.sum(),
  geometry: norwayGeom,
  scale: 10000,
  maxPixels: 1e13,
  bestEffort: true
});

var norwayPopWeightedTemp = ee.Number(sums.get('temp_x_pop'))
  .divide(ee.Number(sums.get('pop')));

print('Norway population-weighted temperature on 2020-01-01:', norwayPopWeightedTemp);

// ===============================
// STEP 6: Build daily population-weighted temperature table
// ===============================

// Number of days
var nDays = ee.Date(endDate).difference(ee.Date(startDate), 'day');

// Create list of dates
var dates = ee.List.sequence(0, nDays.subtract(1)).map(function(day) {
  return ee.Date(startDate).advance(day, 'day');
});

// Convert countries to list
var countryList = nordics.toList(nordics.size());

// Function for one country and one date
var computeDailyCountryTemp = function(date, countryFeature) {
  date = ee.Date(date);
  countryFeature = ee.Feature(countryFeature);

  var countryName = countryFeature.get('country_na');
  var geom = countryFeature.geometry();

  var image = tempC
    .filterDate(date, date.advance(1, 'day'))
    .first();

  var weightedTemp = image.multiply(pop).rename('temp_x_pop');

  var sums = weightedTemp.addBands(pop).reduceRegion({
    reducer: ee.Reducer.sum(),
    geometry: geom,
    scale: 10000,
    maxPixels: 1e13,
    bestEffort: true
  });

  var tempPop = ee.Number(sums.get('temp_x_pop'))
    .divide(ee.Number(sums.get('pop')));

  return ee.Feature(null, {
    'date': date.format('YYYY-MM-dd'),
    'country': countryName,
    'temp_pop_weighted_c': tempPop
  });
};

// Build final feature collection
var resultsList = countryList.map(function(countryFeature) {
  return dates.map(function(date) {
    return computeDailyCountryTemp(date, countryFeature);
  });
}).flatten();

var results = ee.FeatureCollection(resultsList);

// Preview
// Preview only 5 days
var previewDates = ee.List.sequence(0, 4).map(function(day) {
  return ee.Date(startDate).advance(day, 'day');
});

var previewList = countryList.map(function(countryFeature) {
  return previewDates.map(function(date) {
    return computeDailyCountryTemp(date, countryFeature);
  });
}).flatten();

var previewResults = ee.FeatureCollection(previewList);

print('Small preview only:', previewResults);
print('First preview result:', previewResults.first());

// Export for each country 

// ===============================
// FAST VERSION ONLY
// ===============================

var scaleUsed = 25000;  // increase to 50000 if needed

var computeFast = function(image, countryFeature) {
  var countryName = countryFeature.get('country_na');
  var geom = countryFeature.geometry();

  var temp = image.rename('temp_c');
  var weights = pop.rename('weights');

  var weightedMean = temp.addBands(weights).reduceRegion({
    reducer: ee.Reducer.mean().splitWeights(),
    geometry: geom,
    scale: scaleUsed,
    maxPixels: 1e13,
    tileScale: 8,
    bestEffort: true
  });

  return ee.Feature(null, {
    date: image.date().format('YYYY-MM-dd'),
    country: countryName,
    temp_pop_weighted_c: weightedMean.get('mean')
  });
};

var exportCountryFast = function(countryName) {
  var country = nordics.filter(ee.Filter.eq('country_na', countryName)).first();

  var countryResults = ee.FeatureCollection(
    tempC.map(function(img) {
      return computeFast(img, country);
    })
  );

  Export.table.toDrive({
    collection: countryResults,
    description: countryName + '_PopWeighted_Daily_Temperature',
    fileFormat: 'CSV'
  });
};

// ONE country first
exportCountryFast('Sweden');