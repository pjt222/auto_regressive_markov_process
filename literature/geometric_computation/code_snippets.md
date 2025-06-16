# Gray Cuber Code Snippets and Algorithms

## Key Algorithms from The Gray Cuber

### 1. Complex Number Multiplication (Pentagon Generator)
```javascript
// Multiplies two complex numbers represented as [real, imaginary]
function mult(alpha, beta) {
    return [
        alpha[0] * beta[0] - alpha[1] * beta[1],  // Real part
        alpha[0] * beta[1] + alpha[1] * beta[0]   // Imaginary part
    ];
}

// Generate points for n-gon with complex multiplication
function makePoints(vertices = 5, steps = 1, volume = 240) {
    var subvol = round(volume / vertices);
    var point_list = [[[1, 0], [1, 0]]];
    
    for (var vert = 1; vert <= vertices; vert++) {
        // Next vertex on unit circle
        var next = [
            cos(2 * PI * vert * steps / vertices),
            sin(2 * PI * vert * steps / vertices)
        ];
        
        // Linear interpolation in complex plane
        var iter = [
            (-point_list[point_list.length-1][0][0] + next[0]) / subvol,
            (-point_list[point_list.length-1][0][1] + next[1]) / subvol
        ];
        
        // Generate intermediate points
        for (var subpoint = 0; subpoint < subvol; subpoint++) {
            var newPoint = [
                point_list[point_list.length-1][0][0] + iter[0],
                point_list[point_list.length-1][0][1] + iter[1]
            ];
            
            // Apply complex exponentiation
            var result = [1, 0];
            for (expon = 0; expon < vertices - 1; expon++) {
                result = mult(newPoint, result);
            }
            
            point_list.push([newPoint, result]);
        }
    }
    return point_list;
}
```

### 2. 3D Cube Rotation Algorithm
```javascript
// Transform position and rotation based on move axis and direction
function movethem(moveinput, rotateinput) {
    if (moveaxis == 0) {  // X-axis rotation
        return [
            [moveinput[0], -movefact*movedir*moveinput[2], movefact*movedir*moveinput[1]],
            [(1-rotateinput[0])*(1-rotateinput[1]), rotateinput[1]]
        ];
    } else if (moveaxis == 1) {  // Y-axis rotation
        return [
            [movefact*movedir*moveinput[2], moveinput[1], -movefact*movedir*moveinput[0]],
            [rotateinput[0], (1-rotateinput[0])*(1-rotateinput[1])]
        ];
    } else {  // Z-axis rotation
        return [
            [-movefact*movedir*moveinput[1], movefact*movedir*moveinput[0], moveinput[2]],
            [rotateinput[0], rotateinput[1]]
        ];
    }
}

// Smooth animation using cosine interpolation
moveportion = (-cos(PI * moveang / moveframes) + 1) / 4;
```

### 3. Gaussian Prime Detection
```javascript
function makeGauss() {
    for (var re = -wmax; re <= wmax; re++) {
        for (var im = -hmax; im <= hmax; im++) {
            var gaussN = re**2 + im**2;  // Norm of Gaussian integer
            
            var isPrime = false;
            
            // Special cases on axes
            if (re == 0) {
                // Check if |im| is prime of form 4k+3
                if (primes[gaussP] == abs(im) && primes[gaussP] % 4 == 3) {
                    isPrime = true;
                }
            } else if (im == 0) {
                // Check if |re| is prime of form 4k+3
                if (primes[gaussP] == abs(re) && primes[gaussP] % 4 == 3) {
                    isPrime = true;
                }
            } else {
                // General case: check if norm is prime
                if (primes[gaussP] == gaussN) {
                    isPrime = true;
                }
            }
            
            if (isPrime) {
                // Color based on mathematical properties
                var thetacolor = atan2(im, re);  // Angle in complex plane
                var normColor = int(norm / 16) % colorPalette;
                var angleColor = int(60 * thetacolor / PI) % colorPalette;
            }
        }
    }
}
```

### 4. Group Structure Computation
```javascript
class Bubble {
    constructor(index, n, x, y, size) {
        // Compute prime factorization
        var n_factor = prime_factorization(n);
        var n_cycle = [];
        
        // Special handling for powers of 2
        for (var n_fact of n_factor) {
            if (n_fact[0] == 2 && n_fact[1] > 2) {
                n_cycle.push([2, [n_fact[1] - 2, 1]]);
            } else if (n_fact[0] == 2 && n_fact[1] == 2) {
                n_cycle.push([2, [1]]);
            } else if (n_fact[0] != 2 && n_fact[1] > 1) {
                n_cycle.push([n_fact[0], [n_fact[1] - 1]]);
            }
        }
        
        // Compute cycle structure
        this.cycles = [];
        for (n_fact of n_cycle) {
            for (var n_pow = 0; n_pow < n_fact[1].length; n_pow++) {
                if (n_pow == this.cycles.length) {
                    this.cycles.push(n_fact[0] ** n_fact[1][n_pow]);
                } else {
                    this.cycles[n_pow] *= (n_fact[0] ** n_fact[1][n_pow]);
                }
            }
        }
        
        // Calculate order
        this.order = 1;
        for (n_fact of this.cycles) {
            this.order *= n_fact;
        }
    }
    
    // Force-directed graph physics
    push() {
        for (var pusher = 0; pusher < this.index; pusher++) {
            var dist_info = pointDist(this.x, this.y, nodes[pusher].x, nodes[pusher].y);
            if (dist_info[2] < threshold) {
                var push_strength = min(constant / dist_info[2], maxForce);
                this.a[0] -= dist_info[0] * push_strength;
                this.a[1] -= dist_info[1] * push_strength;
                nodes[pusher].a[0] += dist_info[0] * push_strength;
                nodes[pusher].a[1] += dist_info[1] * push_strength;
            }
        }
    }
}
```

### 5. Color Encoding for High-Dimensional Data
```javascript
// Example from Gaussian integers - encoding multiple properties in color
class Gauss {
    constructor(re, im, norm, gauss_rand) {
        this.colorvalue = [];
        
        // Encode norm (magnitude)
        this.colorvalue.push(int(norm/16) % sunlen);
        
        // Encode angle
        var thetacolor = atan2(im, re);
        this.colorvalue.push(int(60*thetacolor/PI) % sunlen);
        
        // Encode combined properties
        this.colorvalue.push((this.colorvalue[1]+this.colorvalue[2]) % sunlen);
        
        // Encode linear combination
        this.colorvalue.push(mod(int((re+im)*2+50), sunlen));
        
        // Encode difference
        this.colorvalue.push(mod((abs(re)-abs(im)+160)*3, sunlen));
        
        // Encode with Perlin noise for smoothness
        this.colorvalue.push(int(noise(re/16+75*gauss_rand, im/16+120*gauss_rand)*sunlen));
    }
}
```

## Connections to N-Dimensional Embeddings

### 1. Rotation in Higher Dimensions
The 3D rotation code can be generalized to n-dimensions using rotation matrices:
```
R(i,j,θ) rotates in the plane spanned by dimensions i and j
For n dimensions, there are n(n-1)/2 possible rotation planes
```

### 2. Complex Numbers as 2D Rotations
The complex multiplication in the pentagon generator is equivalent to 2D rotation and scaling:
```
z₁ * z₂ = |z₁||z₂|e^(i(θ₁+θ₂))
```
This extends to:
- Quaternions (4D rotations)
- Clifford algebras (n-dimensional rotations)

### 3. Force-Directed Embeddings
The physics simulation in groups of units is similar to:
- t-SNE's repulsive forces
- UMAP's attractive/repulsive balance
- Graph embedding algorithms like ForceAtlas2

### 4. Multidimensional Color Encoding
The color encoding schemes demonstrate dimensionality reduction:
- Multiple mathematical properties → 3D color space (RGB)
- Similar to PCA or autoencoders compressing high-dimensional data

### 5. Cyclic Structure Detection
The group cycle computation relates to:
- Periodic patterns in time series
- Fourier analysis in signal processing
- Cyclic features in neural networks