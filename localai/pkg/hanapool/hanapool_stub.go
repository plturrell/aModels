//go:build !hana

package hanapool

import "database/sql"

type Config struct{}

type Pool struct{}

func NewPoolFromEnv() (*Pool, error) { return nil, nil }

func (p *Pool) GetDB() *sql.DB { return nil }

func (p *Pool) Execute(_ interface{}, _ string, _ ...any) (sql.Result, error) {
    return nil, nil
}


