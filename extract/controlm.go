package main

import (
	"encoding/xml"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"
	"unicode"
)

// ControlMJob represents a single job definition in a Control-M XML file.
type ControlMJob struct {
	XMLName        xml.Name  `xml:"JOB"`
	JobName        string    `xml:"JOBNAME,attr"`
	Description    string    `xml:"DESCRIPTION,attr"`
	Command        string    `xml:"COMMAND,attr"`
	RunAs          string    `xml:"RUN_AS,attr"`
	Application    string    `xml:"APPLICATION,attr"`
	SubApplication string    `xml:"SUB_APPLICATION,attr"`
	Host           string    `xml:"HOST,attr"`
	TaskType       string    `xml:"TASKTYPE,attr"`
	CalendarName   string    `xml:"CALENDAR_NAME,attr"`
	Days           string    `xml:"DAYS,attr"`
	WeekDays       string    `xml:"WEEK_DAYS,attr"`
	MonthDays      string    `xml:"MONTH_DAYS,attr"`
	TimeFrom       string    `xml:"TIMEFROM,attr"`
	TimeTo         string    `xml:"TIMETO,attr"`
	MaxRerun       string    `xml:"MAXRERUN,attr"`
	MaxWait        string    `xml:"MAXWAIT,attr"`
	Cyclic         string    `xml:"CYCLIC,attr"`
	Interval       string    `xml:"INTERVAL,attr"`
	When           string    `xml:"WHEN,attr"`
	Priority       string    `xml:"PRIORITY,attr"`
	OrderMethod    string    `xml:"ORDER_METHOD,attr"`
	Folder         string    `xml:"FOLDER,attr"`
	ODate          string    `xml:"ODATE,attr"`
	InConds        []InCond  `xml:"INCOND"`
	OutConds       []OutCond `xml:"OUTCOND"`
}

// InCond represents an input condition for a Control-M job.
type InCond struct {
	Name  string `xml:"NAME,attr"`
	ODate string `xml:"ODATE,attr"`
	Sign  string `xml:"SIGN,attr"`
	AndOr string `xml:"AND_OR,attr"`
}

// OutCond represents an output condition for a Control-M job.
type OutCond struct {
	Name  string `xml:"NAME,attr"`
	ODate string `xml:"ODATE,attr"`
	Sign  string `xml:"SIGN,attr"`
	Type  string `xml:"TYPE,attr"`
}

// parseControlMXML parses a Control-M XML file and returns a slice of ControlMJob structs.
func parseControlMXML(filePath string) ([]ControlMJob, error) {
	xmlFile, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open control-m xml file: %w", err)
	}
	defer xmlFile.Close()

	var jobs []ControlMJob
	// We need a wrapper to handle the root element
	decoder := xml.NewDecoder(xmlFile)
	for {
		t, err := decoder.Token()
		if err != nil {
			break
		}
		if se, ok := t.(xml.StartElement); ok && se.Name.Local == "JOB" {
			var job ControlMJob
			if err := decoder.DecodeElement(&job, &se); err != nil {
				break
			}
			jobs = append(jobs, job)
		}
	}

	return jobs, nil
}

func (job ControlMJob) Properties() map[string]any {
	props := map[string]any{}
	setString(props, "command", job.Command)
	setString(props, "description", job.Description)
	setString(props, "run_as", job.RunAs)
	setString(props, "application", job.Application)
	setString(props, "sub_application", job.SubApplication)
	setString(props, "host", job.Host)
	setString(props, "task_type", job.TaskType)
	setString(props, "folder", job.Folder)
	setString(props, "order_method", job.OrderMethod)
	setString(props, "priority", job.Priority)

	schedule := map[string]any{}
	setString(schedule, "calendar", job.CalendarName)
	setString(schedule, "days", job.Days)
	setString(schedule, "week_days", job.WeekDays)
	setString(schedule, "month_days", job.MonthDays)

	if t := formatControlMTime(job.TimeFrom); t != "" {
		schedule["time_from"] = t
	}
	if t := formatControlMTime(job.TimeTo); t != "" {
		schedule["time_to"] = t
	}

	if val, ok := parseInt(job.MaxRerun); ok {
		schedule["max_rerun"] = val
	}
	if val, ok := parseInt(job.MaxWait); ok {
		schedule["max_wait"] = val
	}

	if strings.EqualFold(strings.TrimSpace(job.Cyclic), "Y") {
		schedule["cyclic"] = true
		if val, ok := parseInt(job.Interval); ok {
			schedule["interval_minutes"] = val
		} else {
			setString(schedule, "interval_raw", job.Interval)
		}
	}

	setString(schedule, "when", job.When)

	if odate := decodeODate(job.ODate); len(odate) > 0 {
		schedule["order_date"] = odate
	}

	if len(schedule) > 0 {
		props["schedule"] = schedule
	}

	return mapOrNil(props)
}

func (c InCond) Properties() map[string]any {
	props := map[string]any{}
	setString(props, "sign", c.Sign)
	setString(props, "and_or", c.AndOr)
	if odate := decodeODate(c.ODate); len(odate) > 0 {
		props["odate"] = odate
	}
	return mapOrNil(props)
}

func (c OutCond) Properties() map[string]any {
	props := map[string]any{}
	setString(props, "sign", c.Sign)
	setString(props, "type", c.Type)
	if odate := decodeODate(c.ODate); len(odate) > 0 {
		props["odate"] = odate
	}
	return mapOrNil(props)
}

func decodeODate(raw string) map[string]any {
	value := strings.TrimSpace(raw)
	if value == "" {
		return nil
	}

	result := map[string]any{
		"raw": value,
	}

	upper := strings.ToUpper(value)
	switch {
	case upper == "ODAT":
		result["type"] = "run_date"
	case upper == "ANY" || upper == "*":
		result["type"] = "any"
	case strings.HasPrefix(upper, "ODAT"):
		result["type"] = "offset"
		result["offset"] = strings.TrimPrefix(upper, "ODAT")
	case len(value) == 8 && isDigits(value):
		result["type"] = "fixed"
		if ts, err := time.Parse("20060102", value); err == nil {
			result["date"] = ts.Format(time.RFC3339)
		}
	default:
		result["type"] = "custom"
	}

	return result
}

func parseInt(raw string) (int, bool) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return 0, false
	}
	val, err := strconv.Atoi(raw)
	if err != nil {
		return 0, false
	}
	return val, true
}

func formatControlMTime(raw string) string {
	raw = strings.TrimSpace(raw)
	if len(raw) != 4 || !isDigits(raw) {
		return ""
	}
	return fmt.Sprintf("%s:%s", raw[:2], raw[2:])
}

func isDigits(value string) bool {
	for _, r := range value {
		if !unicode.IsDigit(r) {
			return false
		}
	}
	return true
}

func setString(props map[string]any, key, value string) {
	value = strings.TrimSpace(value)
	if value == "" {
		return
	}
	props[key] = value
}
