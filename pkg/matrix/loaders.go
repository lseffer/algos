package matrix

import (
	"encoding/csv"
	"log"
	"os"
	"strconv"
)

// ReadCsvFile read a csv file from a path
func ReadCsvFile(filePath string) [][]string {
	f, err := os.Open(filePath)
	if err != nil {
		log.Fatal("Unable to read input file "+filePath, err)
	}
	defer f.Close()

	csvReader := csv.NewReader(f)
	records, err := csvReader.ReadAll()
	if err != nil {
		log.Fatal("Unable to parse file as CSV for "+filePath, err)
	}

	return records
}

func parseMatrix(stringArr [][]string) (res *DenseMatrix, err error) {
	var parsedValue float64
	rows := len(stringArr)
	cols := len(stringArr[0])
	res, err = InitializeMatrix(rows, cols)
	if err != nil {
		return
	}
	for rowIndex, row := range stringArr {
		vec := InitializeVector(cols)
		for colIndex, value := range row {
			parsedValue, err = strconv.ParseFloat(value, 64)
			if err != nil {
				return
			}
			vec.Values[colIndex] = parsedValue
		}
		res.Rows[rowIndex] = vec
	}
	return
}

// ReadCsvFileToMatrix read a csv file and attempt to parse into a matrix
func ReadCsvFileToMatrix(filePath string) (res *DenseMatrix, err error) {
	records := ReadCsvFile(filePath)
	res, err = parseMatrix(records)
	return
}
