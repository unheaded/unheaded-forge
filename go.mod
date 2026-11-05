module unheaded-forge

go 1.26.0

toolchain go1.27.3

require (
	github.com/lib/pq v1.12.3
	unheaded v0.0.0
)

replace unheaded => ../../unheaded

replace github.com/unheaded/doomgeneric => ../doomgeneric/unheaded
