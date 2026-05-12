// MCP Weather Server — Go
//
// Exposes two tools and one resource over the stdio transport:
//
//	Tools:     get_weather(city, units)   — current conditions
//	           get_forecast(city, days)   — multi-day forecast
//	Resources: weather://status           — health and uptime
//
// Usage:
//
//	go run ./weather_mcp_server/
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"hash/fnv"
	"math/rand"
	"os"
	"strings"
	"time"

	"github.com/mark3labs/mcp-go/mcp"
	"github.com/mark3labs/mcp-go/server"
)

// ---------------------------------------------------------------------------
// Mock weather data
// ---------------------------------------------------------------------------

type weatherRecord struct {
	TempC     float64
	Humidity  int
	Condition string
	WindKPH   int
	Country   string
}

var weatherDB = map[string]weatherRecord{
	"tokyo":     {22, 68, "partly cloudy", 14, "JP"},
	"london":    {12, 80, "overcast", 20, "UK"},
	"new york":  {18, 60, "sunny", 12, "US"},
	"paris":     {16, 72, "light rain", 8, "FR"},
	"sydney":    {20, 65, "clear", 18, "AU"},
	"berlin":    {10, 75, "cloudy", 22, "DE"},
	"dubai":     {38, 45, "sunny", 16, "AE"},
	"moscow":    {5, 70, "snow", 10, "RU"},
	"singapore": {30, 85, "thunderstorm", 24, "SG"},
	"toronto":   {8, 62, "clear", 15, "CA"},
}

var forecastConditions = []string{
	"sunny", "partly cloudy", "cloudy", "light rain",
	"rain", "thunderstorm", "clear", "overcast",
}

func normalizeCity(city string) string {
	parts := strings.SplitN(city, ",", 2)
	return strings.ToLower(strings.TrimSpace(parts[0]))
}

func cToF(c float64) float64 {
	return float64(int((c*9/5+32)*10)) / 10
}

// seededRand returns a deterministic *rand.Rand for the given city.
func seededRand(city string) *rand.Rand {
	h := fnv.New64a()
	_, _ = h.Write([]byte(city))
	return rand.New(rand.NewSource(int64(h.Sum64()))) //nolint:gosec
}

func listSupportedCities() []string {
	cities := make([]string, 0, len(weatherDB))
	for k := range weatherDB {
		cities = append(cities, strings.Title(k)) //nolint:staticcheck
	}
	return cities
}

// ---------------------------------------------------------------------------
// Tool handlers
// ---------------------------------------------------------------------------

func handleGetWeather(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	city, _ := req.Params.Arguments["city"].(string)
	if city == "" {
		return mcp.NewToolResultText(`{"error":"Missing required argument: city"}`), nil
	}
	units, _ := req.Params.Arguments["units"].(string)
	if units == "" {
		units = "celsius"
	}

	fmt.Fprintf(os.Stderr, "[weather-server] get_weather: city=%q units=%s\n", city, units)

	rec, ok := weatherDB[normalizeCity(city)]
	if !ok {
		payload, _ := json.Marshal(map[string]any{
			"error":            fmt.Sprintf("City not found: %q", city),
			"supported_cities": listSupportedCities(),
		})
		return mcp.NewToolResultText(string(payload)), nil
	}

	temp := rec.TempC
	if units == "fahrenheit" {
		temp = cToF(rec.TempC)
	}

	data := map[string]any{
		"city":        city,
		"country":     rec.Country,
		"temperature": temp,
		"units":       units,
		"humidity":    rec.Humidity,
		"condition":   rec.Condition,
		"wind_kph":    rec.WindKPH,
		"timestamp":   time.Now().UTC().Format(time.RFC3339),
		"source":      "mock_data",
	}
	payload, _ := json.MarshalIndent(data, "", "  ")
	return mcp.NewToolResultText(string(payload)), nil
}

func handleGetForecast(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	city, _ := req.Params.Arguments["city"].(string)
	if city == "" {
		return mcp.NewToolResultText(`{"error":"Missing required argument: city"}`), nil
	}
	days := 5
	if d, ok := req.Params.Arguments["days"].(float64); ok {
		days = int(d)
	}
	if days < 1 {
		days = 1
	}
	if days > 10 {
		days = 10
	}

	fmt.Fprintf(os.Stderr, "[weather-server] get_forecast: city=%q days=%d\n", city, days)

	rec, ok := weatherDB[normalizeCity(city)]
	if !ok {
		payload, _ := json.Marshal(map[string]any{
			"error":            fmt.Sprintf("City not found: %q", city),
			"supported_cities": listSupportedCities(),
		})
		return mcp.NewToolResultText(string(payload)), nil
	}

	rng := seededRand(normalizeCity(city))
	forecastDays := make([]map[string]any, days)
	for i := range forecastDays {
		date := time.Now().UTC().AddDate(0, 0, i+1)
		forecastDays[i] = map[string]any{
			"date":                     date.Format("2006-01-02"),
			"day":                      date.Weekday().String()[:3],
			"high_c":                   float64(int((rec.TempC+rng.Float64()*4+1)*10)) / 10,
			"low_c":                    float64(int((rec.TempC-rng.Float64()*4-1)*10)) / 10,
			"condition":                forecastConditions[rng.Intn(len(forecastConditions))],
			"precipitation_chance_pct": rng.Intn(100),
		}
	}

	data := map[string]any{
		"city":     city,
		"days":     days,
		"forecast": forecastDays,
		"source":   "mock_data",
	}
	payload, _ := json.MarshalIndent(data, "", "  ")
	return mcp.NewToolResultText(string(payload)), nil
}

func handleStatus(ctx context.Context, req mcp.ReadResourceRequest) ([]mcp.ResourceContents, error) {
	status := map[string]any{
		"status":           "healthy",
		"server":           "weather-server",
		"version":          "1.0.0",
		"tools":            []string{"get_weather", "get_forecast"},
		"supported_cities": listSupportedCities(),
		"timestamp":        time.Now().UTC().Format(time.RFC3339),
		"transport":        "stdio",
	}
	payload, err := json.MarshalIndent(status, "", "  ")
	if err != nil {
		return nil, err
	}
	return []mcp.ResourceContents{
		mcp.TextResourceContents{
			URI:      req.Params.URI,
			MIMEType: "application/json",
			Text:     string(payload),
		},
	}, nil
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

func main() {
	s := server.NewMCPServer("weather-server", "1.0.0",
		server.WithToolCapabilities(false),
		server.WithResourceCapabilities(true, false),
	)

	// --- Tools ---
	weatherTool := mcp.NewTool("get_weather",
		mcp.WithDescription(
			"Get current weather conditions for a city. Returns temperature, "+
				"humidity, wind speed, and conditions. "+
				"Examples: 'Tokyo, JP', 'London, UK', 'New York, US'.",
		),
		mcp.WithString("city",
			mcp.Required(),
			mcp.Description(
				"City name with optional ISO country code. "+
					"Examples: 'Tokyo, JP', 'Sydney, AU'.",
			),
		),
		mcp.WithString("units",
			mcp.Description("Temperature unit: 'celsius' (default) or 'fahrenheit'."),
			mcp.Enum("celsius", "fahrenheit"),
		),
	)
	s.AddTool(weatherTool, handleGetWeather)

	forecastTool := mcp.NewTool("get_forecast",
		mcp.WithDescription(
			"Get a multi-day weather forecast for a city (1–10 days). "+
				"Returns daily high/low temperatures, conditions, and precipitation chance.",
		),
		mcp.WithString("city",
			mcp.Required(),
			mcp.Description("City name with optional ISO country code."),
		),
		mcp.WithNumber("days",
			mcp.Description("Number of forecast days (1–10). Defaults to 5."),
		),
	)
	s.AddTool(forecastTool, handleGetForecast)

	// --- Resource ---
	statusResource := mcp.NewResource(
		"weather://status",
		"Server Status",
		mcp.WithResourceDescription(
			"Health, uptime, and capability information for this weather MCP server.",
		),
		mcp.WithMIMEType("application/json"),
	)
	s.AddResource(statusResource, handleStatus)

	// --- Start ---
	fmt.Fprintln(os.Stderr, "[weather-server] Starting on stdio transport ...")
	if err := server.ServeStdio(s); err != nil {
		fmt.Fprintf(os.Stderr, "[weather-server] Fatal: %v\n", err)
		os.Exit(1)
	}
}
